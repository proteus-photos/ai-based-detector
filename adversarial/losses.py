import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_acc(logits, targets):
    preds_clean = logits.max(dim=1)[1].detach()
    acc = (preds_clean.eq(targets).sum() / targets.shape[0]).item() * 100
    return acc


def compute_loss(
    loss_str,
    embedding,
    targets,
    embedding_orig,
    logit_scale,
    embedding_text_labels_norm=None,
    reduction="mean",
):
    if loss_str == "l2":
        loss = l2(out=embedding, targets=embedding_orig, reduction=reduction)
    elif loss_str == "ce":
        loss = ce(
            out=embedding @ (logit_scale * embedding_text_labels_norm),
            targets=targets,
            reduction=reduction,
        )
    else:
        raise ValueError(f"loss {loss_str} not supported")
    return loss


def l2(out, targets, reduction="none"):
    # squared l2 - it does not divide by the latent dimension
    # should have shape (batch_size, embedding_size)
    assert out.shape == targets.shape, f"{out.shape} != {targets.shape}"
    assert out.shape[0] > 1
    # Compute the element-wise squared error
    squared_error_batch = F.mse_loss(out, targets, reduction="none")
    if reduction == "mean":
        squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    else:
        squared_error_batch = squared_error_batch.sum(dim=1)
        assert squared_error_batch.shape == (
            out.shape[0],
        ), f"{squared_error_batch.shape} != {(out.shape[0],)}"
    return squared_error_batch


def ce(out, targets, reduction="mean"):
    # out = logits
    assert out.shape[0] == targets.shape[0], (out.shape, targets.shape)
    assert out.shape[0] > 1

    return F.cross_entropy(out, targets, reduction=reduction)


class ComputeLossWrapper:
    def __init__(
        self,
        embedding_orig,
        embedding_text_labels_norm,
        reduction="mean",
        loss=None,
        logit_scale=100.0,
    ):
        self.embedding_orig = embedding_orig
        self.embedding_text_labels_norm = embedding_text_labels_norm
        self.reduction = reduction
        self.loss_str = loss
        self.logit_scale = logit_scale

    def __call__(self, embedding, targets):
        return compute_loss(
            loss_str=self.loss_str,
            embedding=embedding,
            targets=targets,
            embedding_orig=self.embedding_orig,
            logit_scale=self.logit_scale,
            embedding_text_labels_norm=self.embedding_text_labels_norm,
            reduction=self.reduction,
        )
