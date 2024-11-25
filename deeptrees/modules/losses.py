import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from .metrics import iou, iou_with_logits


class BinarySegmentationLoss(_Loss):
    """Combines binary cross entropy loss with -log(iou).
    Works with probabilities, so after applying sigmoid activation."""

    def __init__(self, iou_weight=0.5, **kwargs):
        super().__init__()
        self.bceloss = nn.BCELoss(**kwargs)
        self.iou_weight = iou_weight

    def forward(self, y_pred, y_true):
        loss = (1 - self.iou_weight) * self.bceloss(y_pred, y_true)
        loss -= self.iou_weight * torch.log(iou(y_pred, y_true))
        return loss

class BinarySegmentationLossWithLogits(_Loss):
    """Combines binary cross entropy loss with -log(iou).
    Works with logits - don't apply sigmoid to your network output."""

    def __init__(self, iou_weight=0.5, **kwargs):
        super().__init__()
        self.bceloss = nn.BCEWithLogitsLoss(**kwargs)
        self.iou_weight = iou_weight

    def forward(self, y_pred, y_true):
        loss = (1 - self.iou_weight) * self.bceloss(y_pred, y_true)
        loss -= self.iou_weight * torch.log(iou_with_logits(y_pred, y_true))
        return loss
