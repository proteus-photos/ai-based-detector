import torch
import torch.nn.functional as F


def ce(out, targets, reduction="none"):
    # out = logits
    assert out.shape[0] == targets.shape[0], (out.shape, targets.shape)
    assert out.shape[0] > 1
    targets = targets.float()
    return F.binary_cross_entropy(torch.sigmoid(out), targets, reduction=reduction)


def hinge_loss(outputs, labels):
    return torch.clamp(1 - outputs * labels, min=0)
