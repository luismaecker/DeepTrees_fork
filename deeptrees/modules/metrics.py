import torch


def iou(y_pred, y_true):
    """
    Calculate the Intersection over Union (IoU) between two tensors.
    The IoU is a measure of the overlap between two sets, defined as the 
    intersection divided by the union of the sets. It is commonly used 
    in image segmentation tasks to evaluate the accuracy of predictions.
    Args:
        y_pred (torch.Tensor): Predicted tensor, typically a binary mask.
        y_true (torch.Tensor): Ground truth tensor, typically a binary mask.
    Returns:
        float: The IoU score, a value between 0 and 1, where 1 indicates 
               perfect overlap and 0 indicates no overlap.
    Note:
        - The function uses a small epsilon value to avoid division by zero.
        - Both input tensors should have the same shape.
        - The tensors are expected to be of type torch.float32.
    """
    
    eps = torch.finfo(torch.float32).eps  # A more stable epsilon for float32
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return (intersection + eps) / (union + eps)


def iou_with_logits(y_pred, y_true):
    """
    Compute the Intersection over Union (IoU) score with logits.
    This function applies a sigmoid activation to the predicted logits and then
    calculates the IoU score between the predicted and true values.
    Args:
        y_pred (torch.Tensor): The predicted logits tensor. This tensor should contain raw, unnormalized scores.
        y_true (torch.Tensor): The ground truth binary tensor. This tensor should contain binary values (0 or 1).
    Returns:
        float: The IoU score between the predicted and true values.
    Example:
        >>> y_pred = torch.tensor([[0.8, 0.4], [0.3, 0.9]])
        >>> y_true = torch.tensor([[1, 0], [0, 1]])
        >>> iou_score = iou_with_logits(y_pred, y_true)
        >>> print(iou_score)
    """
    
    output = torch.sigmoid(y_pred)
    return iou(output, y_true)
