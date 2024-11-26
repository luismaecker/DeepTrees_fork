import torch
import segmentation_models_pytorch as smp

import lightning as L
from deeptrees.modules import utils
from deeptrees.modules import metrics
from deeptrees.modules.losses import BinarySegmentationLossWithLogits


class SegmentationModel(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 4,
        architecture: str = "Unet",
        backbone: str = "resnet18",
        lr: float = 1e-4,
        mask_loss_share: float = 0.5,
    ):
        """
        Segmentation model

        A segmentation model which takes an input image and returns a foreground / background mask along with object
        outlines.

        Args:
            in_channels (int): Number of input channels
            architecture (str): One of 'Unet, Unet++, Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3+'
            backbone (str): One of the backbones supported by the [pytorch segmentation models package](https://github.com/qubvel/segmentation_models.pytorch)
            lr (float): Learning rate
            mask_loss_share (float): Value between 0 and 1. Zero means only the outlines contribute to the loss,
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
        self.seg_loss = BinarySegmentationLossWithLogits(reduction="mean")
        # self.focal_loss = BinaryFocalLossWithLogits()
        self.lr = lr
        self.mask_loss_share = mask_loss_share
        self.save_hyperparameters()  # logs the arguments of this function

    def forward(self, x):
        """
        Perform a forward pass through the segmentation model.
        Args:
            x (torch.Tensor): Input tensor to the model.
        Returns:
            torch.Tensor: Output tensor from the model after the forward pass.
        """

        return self.model(x)

    def shared_step(self, batch):
        """
        Perform a shared step in the model training/validation process.
        Args:
            batch (tuple): A tuple containing input data (x) and target data (y).
                           - x (Tensor): Input tensor to the model.
                           - y (Tensor): Target tensor containing the true masks and outlines.
                             - y[:, 0]: True tree cover mask.
                             - y[:, 1]: True outlines.
        Returns:
            tuple: A tuple containing the following elements:
                - loss (Tensor): Combined loss calculated from mask and outline losses.
                - loss_mask (Tensor): Loss calculated for the tree cover mask.
                - loss_outline (Tensor): Loss calculated for the outlines.
                - iou_mask (Tensor): Intersection over Union (IoU) metric for the tree cover mask.
                - iou_outline (Tensor): Intersection over Union (IoU) metric for the outlines.
        """

        x, y = batch
        # t for true, p for prediction
        mask_t = y[:, 0]  # the first layer is the tree cover mask
        outline_t = y[
            :, 1
        ]  # the second are the outlines, optionally there could be more than these two

        res = self.model(x)
        mask_p = res[:, 0]
        outline_p = res[:, 1]

        # calculate iou metric
        iou_mask = metrics.iou_with_logits(mask_p, mask_t)
        iou_outline = metrics.iou_with_logits(outline_p, outline_t)

        loss_mask = self.seg_loss(mask_p, mask_t)
        # loss_outline = 100 * self.focal_loss(outline_p, outline_t) - torch.log(iou_outline)
        loss_outline = self.seg_loss(outline_p, outline_t)
        loss = (
            self.mask_loss_share * loss_mask + (1 - self.mask_loss_share) * loss_outline
        )

        # FIXME mask_loss_share is actually not used when using the combined model

        return loss, loss_mask, loss_outline, iou_mask, iou_outline

    @torch.jit.ignore
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        Args:
            batch (Tensor): The input batch of data.
            batch_idx (int): The index of the batch.
        Returns:
            Tensor: The computed loss for the batch.
        Logs:
            train/loss (float): The overall training loss.
            train/loss_mask (float): The training loss for the mask.
            train/loss_outline (float): The training loss for the outline.
            train/iou_mask (float): The Intersection over Union (IoU) for the mask.
            train/iou_outline (float): The Intersection over Union (IoU) for the outline.
        """

        loss, loss_mask, loss_outline, iou_mask, iou_outline = self.shared_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "train/loss_mask", loss_mask, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log(
            "train/loss_outline",
            loss_outline,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/iou_mask", iou_mask, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log(
            "train/iou_outline",
            iou_outline,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    @torch.jit.ignore
    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        Args:
            batch (tuple): A batch of data from the validation dataset.
            batch_idx (int): The index of the batch.
        Returns:
            torch.Tensor: The computed loss for the batch.
        Logs:
            val/loss (float): The overall validation loss.
            val/loss_mask (float): The validation loss for the mask.
            val/loss_outline (float): The validation loss for the outline.
            val/iou_mask (float): The Intersection over Union (IoU) for the mask.
            val/iou_outline (float): The Intersection over Union (IoU) for the outline.
        """
        
        loss, loss_mask, loss_outline, iou_mask, iou_outline = self.shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "val/loss_mask", loss_mask, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log(
            "val/loss_outline",
            loss_outline,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("val/iou_mask", iou_mask, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "val/iou_outline", iou_outline, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for the model.
        This method sets up the Adam optimizer with the learning rate specified by `self.lr`.
        It also sets up a cosine annealing warm restarts scheduler with a period of 30 epochs
        and 2 restarts.
        Returns:
            tuple: A tuple containing a list with the optimizer and a list with the scheduler.
        """
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 30, 2
        )
        return [optimizer], [scheduler]
