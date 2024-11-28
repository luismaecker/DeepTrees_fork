import torch

class InferenceModel(torch.nn.Module):
    """Just a wrapper to apply the sigmoid activation to mask and outlines during inference.."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Perform a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor to the model.
        Returns:
            torch.Tensor or tuple: If the model output contains two elements, returns a tuple (y, metric) where y is the 
            output tensor with sigmoid activation applied to the first two columns, and metric is the second element of 
            the output. If the model output contains only one element, returns y with sigmoid activation applied to the 
            first two columns.
        """
        
        output = self.model(x)
        if len(output) == 2:
            y, metric = output
            y[:,:2] = torch.sigmoid(y[:,:2])
            return y, metric
        else:
            y = output
            y[:,:2] = torch.sigmoid(y[:,:2])
            return y
