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
        """
        Computes the custom loss function which is a combination of Binary Cross-Entropy (BCE) loss and 
        Intersection over Union (IoU) loss.
        Parameters:
        -----------
        y_pred : torch.Tensor
            The predicted output tensor from the model. It should have the same shape as `y_true`.
        y_true : torch.Tensor
            The ground truth tensor. It should have the same shape as `y_pred`.
        Returns:
        --------
        torch.Tensor
            The computed loss value which is a weighted sum of BCE loss and the negative logarithm of IoU.
        Notes:
        ------
        - The BCE loss is weighted by `(1 - self.iou_weight)`.
        - The IoU loss is weighted by `self.iou_weight` and is computed as the negative logarithm of the IoU.
        - Ensure that `iou` function is defined and computes the Intersection over Union correctly.
        """
        
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
        """
        Computes the loss by combining Binary Cross-Entropy (BCE) loss and Intersection over Union (IoU) loss.
        Args:
            y_pred (torch.Tensor): The predicted output tensor from the model. This tensor typically contains
                                   the predicted probabilities for each class.
            y_true (torch.Tensor): The ground truth tensor. This tensor contains the actual class labels.
        Returns:
            torch.Tensor: The computed loss value which is a combination of BCE loss and IoU loss.
        The loss is calculated as follows:
        1. Compute the BCE loss between the predicted and true values.
        2. Compute the IoU loss between the predicted and true values.
        3. Combine the two losses using the `iou_weight` attribute to balance their contributions.
        Note:
            - The `iou_weight` attribute should be defined in the class to control the balance between BCE and IoU losses.
            - The `bceloss` method should be defined in the class to compute the BCE loss.
            - The `iou_with_logits` function should be defined to compute the IoU loss with logits.
        """
        
        loss = (1 - self.iou_weight) * self.bceloss(y_pred, y_true)
        loss -= self.iou_weight * torch.log(iou_with_logits(y_pred, y_true))
        return loss
