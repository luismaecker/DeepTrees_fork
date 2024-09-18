import torch


def iou(y_pred, y_true):
    eps = torch.finfo(torch.float32).eps  # A more stable epsilon for float32
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return (intersection + eps) / (union + eps)


def iou_with_logits(y_pred, y_true):
    output = torch.sigmoid(y_pred)
    return iou(output, y_true)
