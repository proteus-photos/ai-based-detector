import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from adversarial.utils import project_perturbation, normalize_grad

# from fra31/robust-finetuning

def L1_norm(x, keepdim=False):
    z = x.abs().view(x.shape[0], -1).sum(-1)
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L0_norm(x):
    return (x != 0.).view(x.shape[0], -1).sum(-1)

def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 = eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    # u = torch.min(u, epsinf - torch.clone(y).abs())
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()

    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)

    inu = 2* (indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)

    s1 = -u.sum(dim=1)

    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)

    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)
    # print(s[0])

    # print(c5.shape, c2)

    if c2.nelement != 0:

        lb = torch.zeros_like(c2).float()
        ub = torch.ones_like(lb) * (bs.shape[1] - 1)

        # print(c2.shape, lb.shape)

        nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
        counter2 = torch.zeros_like(lb).long()
        counter = 0

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2.)
            counter2 = counter4.type(torch.LongTensor)

            c8 = s[c2, counter2] + c[c2] < 0
            ind3 = c8.nonzero().squeeze(1)
            ind32 = (~c8).nonzero().squeeze(1)
            # print(ind3.shape)
            if ind3.nelement != 0:
                lb[ind3] = counter4[ind3]
            if ind32.nelement != 0:
                ub[ind32] = counter4[ind32]

            # print(lb, ub)
            counter += 1

        lb2 = lb.long()
        alpha = (-s[c2, lb2] - c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
        d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])

    return (sigma * d).view(x2.shape)


def dlr_loss(x, y, reduction='none'):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    return -(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - \
             x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)


def dlr_loss_targeted(x, y, y_target):
    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (
            x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)


# criterion_dict = {
#     'ce': lambda x, y: F.cross_entropy(x, y, reduction='none'),
#     'dlr': dlr_loss, 'dlr-targeted': dlr_loss_targeted
# }


def check_oscillation(x, j, k, y5, k3=0.75):
    t = torch.zeros(x.shape[1]).to(x.device)
    for counter5 in range(k):
        t += (x[j - counter5] > x[j - counter5 - 1]).float()

    return (t <= k * k3 * torch.ones_like(t)).float()


def apgd_train(model, x, y, norm, eps, n_iter=10, use_rs=False, loss_fn=None,
               verbose=False, is_train=True, initial_stepsize=None):

    print(model.training)
    assert not model.training
    norm = norm.replace('linf', 'Linf').replace('l2', 'L2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ndims = len(x.shape) - 1

    if not use_rs:
        x_adv = x.clone()
    else:
        raise NotImplemented
        if norm == 'Linf':
            t = torch.rand_like(x)

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]], device=device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]], device=device)
    # acc_steps = torch.zeros_like(loss_best_steps)  # Commented out

    n_fts = math.prod(x.shape[1:])
    if norm in ['Linf', 'L2']:
        n_iter_2 = max(int(0.22 * n_iter), 1)
        n_iter_min = max(int(0.06 * n_iter), 1)
        size_decr = max(int(0.03 * n_iter), 1)
        k = n_iter_2 + 0
        thr_decr = .75
        alpha = 2.
    elif norm in ['L1']:
        k = max(int(.04 * n_iter), 1)
        init_topk = .05 if is_train else .2
        topk = init_topk * torch.ones([x.shape[0]], device=device)
        sp_old = n_fts * torch.ones_like(topk)
        adasp_redstep = 1.5
        adasp_minstep = 10.
        alpha = 1.

    if initial_stepsize:
        alpha = initial_stepsize / eps

    step_size = alpha * eps * torch.ones(
        [x.shape[0], *[1] * ndims],
        device=device
    )
    counter3 = 0

    x_adv.requires_grad_()
    logits = F.normalize(model(x_adv), dim=-1)
    loss_indiv = loss_fn(logits, y).to(device)
    loss = loss_indiv.sum().to(device)
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    grad_best = grad.clone()
    x_adv.detach_()
    loss_indiv.detach_()
    loss.detach_()

    # print(logits.detach())
    # acc = logits.detach().max(1)[1] == y  # Commented out
    # acc_steps[0] = acc + 0  # Commented out
    loss_best = loss_indiv.detach().clone()
    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    n_reduced = 0

    u = torch.arange(x.shape[0], device=device)
    x_adv_old = x_adv.clone().detach()

    for i in range(n_iter):
        x_adv = x_adv.detach()
        grad2 = x_adv - x_adv_old
        x_adv_old = x_adv.clone()
        loss_curr = loss.detach().mean()

        a = 0.75 if i > 0 else 1.0

        if norm == 'Linf':
            x_adv_1 = x_adv + step_size * torch.sign(grad)
            x_adv_1 = torch.clamp(
                torch.min(
                    torch.max(x_adv_1, x - eps), x + eps
                ), 0.0, 1.0
            )
            x_adv_1 = torch.clamp(
                torch.min(
                    torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - eps), x + eps
                ), 0.0, 1.0
            )
        
        x_adv = x_adv_1 + 0.

        x_adv.requires_grad_()
        logits = F.normalize(model(x_adv), dim=-1)
        loss_indiv = loss_fn(logits, y)
        loss = loss_indiv.sum()

        if i < n_iter - 1:
            grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        x_adv.detach_()
        loss_indiv.detach_()
        loss.detach_()

        # pred = logits.detach().max(1)[1] == y  # Commented out
        # acc = torch.min(acc, pred)  # Commented out
        # acc_steps[i + 1] = acc + 0  # Commented out
        logits_bin = torch.sign(logits.detach())
        logits_bin[logits_bin==0]=1
        ind_pred = (logits_bin == y).nonzero().squeeze()
        x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
        
        if verbose:
            print(
                'iteration: {} - best loss: {:.6f} curr loss {:.6f}'.format(
                    i, loss_best.sum(), loss_curr
                )
            )

        y1 = loss_indiv.detach().clone().to(device)
        loss_steps[i] = y1 + 0
        ind = (y1 > loss_best).nonzero().squeeze()


        x_best[ind] = x_adv[ind].clone()
        grad_best[ind] = grad[ind].clone()
        loss_best[ind] = y1[ind] + 0
        loss_best_steps[i + 1] = loss_best + 0

        counter3 += 1

        if counter3 == k:
            if norm in ['Linf', 'L2']:
                loss_steps = loss_steps.to(device)
                loss_best = loss_best.to(device)

                fl_oscillation = check_oscillation(loss_steps, i, k, loss_best, k3=thr_decr).to(device)
                fl_reduce_no_impr = (1. - reduced_last_check.to(device)) * (loss_best_last_check.to(device) >= loss_best).float()
                fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)

                reduced_last_check = fl_oscillation.clone().to(device)
                loss_best_last_check = loss_best.clone().to(device)

                if fl_oscillation.sum() > 0:
                    ind_fl_osc = (fl_oscillation > 0).nonzero(as_tuple=False).squeeze().to(device)

                    step_size = step_size.to(device)
                    step_size[ind_fl_osc] /= 2.0
                    n_reduced = fl_oscillation.sum()

                    x_adv = x_adv.to(device)
                    x_best = x_best.to(device)
                    grad_best = grad_best.to(device)

                    x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                    grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                counter3 = 0
                k = max(k - size_decr, n_iter_min)

    return x_best_adv



def pgd(
        forward,
        loss_fn,
        data_clean,
        targets,
        norm,
        eps,
        iterations,
        stepsize,
        output_normalize,
        perturbation=None,
        mode='min',
        momentum=0.9,
        verbose=False
):
    """
    Minimize or maximize given loss
    """
    # make sure data is in image space
    assert torch.max(data_clean) < 1. + 1e-6 and torch.min(data_clean) > -1e-6

    if perturbation is None:
        perturbation = torch.zeros_like(data_clean, requires_grad=True)
    velocity = torch.zeros_like(data_clean)
    for i in range(iterations):
        perturbation.requires_grad = True
        with torch.enable_grad():
            out = forward(data_clean + perturbation, output_normalize=output_normalize)
            loss = loss_fn(out, targets)
            if verbose:
                print(f'[{i}] {loss.item():.5f}')

        with torch.no_grad():
            gradient = torch.autograd.grad(loss, perturbation)[0]
            gradient = gradient
            if gradient.isnan().any():  #
                print(f'attention: nan in gradient ({gradient.isnan().sum()})')  #
                gradient[gradient.isnan()] = 0.
            # normalize
            gradient = normalize_grad(gradient, p=norm)
            # momentum
            velocity = momentum * velocity + gradient
            velocity = normalize_grad(velocity, p=norm)
            # update
            if mode == 'min':
                perturbation = perturbation - stepsize * velocity
            elif mode == 'max':
                perturbation = perturbation + stepsize * velocity
            else:
                raise ValueError(f'Unknown mode: {mode}')
            # project
            perturbation = project_perturbation(perturbation, eps, norm)
            perturbation = torch.clamp(
                data_clean + perturbation, 0, 1
            ) - data_clean  # clamp to image space
            assert not perturbation.isnan().any()
            assert torch.max(data_clean + perturbation) < 1. + 1e-6 and torch.min(
                data_clean + perturbation
            ) > -1e-6

            # assert (ctorch.compute_norm(perturbation, p=self.norm) <= self.eps + 1e-6).all()
    # todo return best perturbation
    # problem is that model currently does not output expanded loss
    return data_clean + perturbation.detach()

