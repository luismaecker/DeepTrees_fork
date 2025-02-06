import segmentation_models_pytorch as smp

import lightning as L
from ..modules import utils


class SegmentationModel(L.LightningModule):
    def __init__(self,
                 in_channels: int = 4,
                 architecture: str = "Unet",
                 backbone: str = "resnet18"):
        """
        Segmentation model

        A segmentation model which takes an input image and returns a foreground / background mask along with object
        outlines.

        Args:
            in_channels (int): Number of input channels
            architecture (str): One of 'Unet, Unet++, Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3+'
            backbone (str): One of the backbones supported by the [pytorch segmentation models package](https://github.com/qubvel/segmentation_models.pytorch)
            one means only the mask loss is relevant. Linear in between.
        """

        super().__init__()

        # architectures should be static
        match architecture:
            case "Unet":
                arch = smp.Unet
            case "Unet++":
                arch = smp.UnetPlusPlus
            case "Linknet":
                arch = smp.Linknet
            case "FPN":
                arch = smp.FPN
            case "PSPNet":
                arch = smp.PSPNet
            case "PAN":
                arch = smp.PAN
            case "DeepLabV3":
                arch = smp.DeepLabV3
            case "DeepLabV3+":
                arch = smp.DeepLabV3Plus
            case _:
                raise ValueError(f"Unsupported architecture: {architecture}")

        self.model = arch(in_channels=in_channels, classes=2, encoder_name=backbone)
        # set batchnorm momentum to tensorflow standard, which works better
        utils.set_batchnorm_momentum(self.model, 0.99)

    def forward(self, x):
        return self.model(x)
