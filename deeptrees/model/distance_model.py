import torch
import segmentation_models_pytorch as smp
import lightning as L
from ..modules import utils

class DistanceModel(L.LightningModule):
    def __init__(self, in_channels: int, architecture: str = "Unet", backbone: str = "resnet18"):
        """ Distance transform model

        The model is the second part in the tree crown delineation model.

        Args:
            in_channels (int): Number of input channels
            architecture (str): One of 'Unet, Unet++, Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3+'
            backbone (str): One of the backbones supported by the [pytorch segmentation models package](https://github.com/qubvel/segmentation_models.pytorch)
        """
        super().__init__()

        # architectures should be static
        match architecture:
            case 'Unet':
                arch = smp.Unet
            case 'Unet++':
                arch = smp.UnetPlusPlus
            case 'Linknet':
                arch = smp.Linknet
            case 'FPN':
                arch = smp.FPN
            case 'PSPNet':
                arch = smp.PSPNet
            case 'PAN':
                arch = smp.PAN
            case 'DeepLabV3':
                arch = smp.DeepLabV3
            case 'DeepLabV3+':
                arch = smp.DeepLabV3Plus

        self.model = arch(encoder_name=backbone,
                          in_channels=in_channels,
                          classes=1,
                          encoder_depth=3,
                          decoder_channels=[64, 32, 16],
                          activation="sigmoid")
        # throw away unused weights
        self.model.encoder.layer3 = None
        self.model.encoder.layer4 = None
        utils.set_batchnorm_momentum(self.model, 0.99)

    def forward(self, img: torch.Tensor, mask_and_outline: torch.Tensor, from_logits: bool = False):
        """ Distance transform forward pass

        Args:
            img (torch.Tensor): Input image
            mask_and_outline (torch.Tensor): Tensor containing mask and outlines concatenated in channel dimension, \
                coming from the first sub-network.
            from_logits (bool): If set to true, sigmoid activation is applied to the mask_and_outline tensor.

        Returns:
            Model output of dimension N1HW
        """
        if from_logits:
            mask_and_outline = torch.sigmoid(mask_and_outline)

        x = torch.cat((img, mask_and_outline), dim=1)
        return self.model(x)
