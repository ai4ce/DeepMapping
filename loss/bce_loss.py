import torch
import torch.nn as nn


class BCEWithLogitsLoss2(nn.Module):
    def __init__(self, weight=None, reduction='elementwise_mean'):
        super(BCEWithLogitsLoss2, self).__init__()
        self.reduction = reduction
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return bce_with_logits(input, target, weight=self.weight, reduction=self.reduction)


def bce_with_logits(input, target, weight=None, reduction='elementwise_mean'):
    """
    This function differs from F.binary_cross_entropy_with_logits in the way 
    that if weight is not None, the loss is normalized by weight
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))
    if weight is not None:
        if not (weight.size() == input.size()):
            raise ValueError("Weight size ({}) must be the same as input size ({})".format(
                weight.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + \
        ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if reduction == 'none':
        return loss
    elif reduction == 'elementwise_mean':
        if weight is not None:
            # different from F.binary_cross_entropy_with_logits
            return loss.sum() / weight.sum()
        else:
            return loss.mean()
    else:
        return loss.sum()


def bce(pred, targets, weight=None):
    criternion = BCEWithLogitsLoss2(weight=weight)
    loss = criternion(pred, targets)
    return loss
