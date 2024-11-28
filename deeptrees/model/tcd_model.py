import rootutils
import os

# path = rootutils.find_root(search_from=__file__, indicator=".project-root")
path = rootutils.find_root(search_from=os.path.abspath(""), indicator=".project-root")

# set root directory
rootutils.set_root(
    path=path,  # path to the root directory
    project_root_env_var=True,  # set the PROJECT_ROOT environment variable to root directory
    dotenv=True,  # load environment variables from .env if exists in root directory
    pythonpath=True,  # add root directory to the PYTHONPATH (helps with imports)
    cwd=False,  # we do not want that with hydra
)

import time

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics.functional.segmentation import mean_iou

from deeptrees.model.segmentation_model import SegmentationModel
from deeptrees.model.distance_model import DistanceModel
from deeptrees.modules import metrics
from deeptrees.modules.losses import BinarySegmentationLossWithLogits
from deeptrees.modules import postprocessing as tcdpp

import logging

log = logging.getLogger(__name__)


class TreeCrownDelineationModel(L.LightningModule):
    def __init__(
        self,
        in_channels,
        architecture="Unet",
        backbone="resnet18",
        lr=1e-4,
        mask_iou_share=0.5,
        apply_sigmoid=False,
        freeze_layers=False,
        track_running_stats=True,
        postprocessing_config={},
    ):
        """Tree crown delineation model

        The model consists of two sub-netoworks (two U-Nets with ResNet backbone). The first network calculates a tree
        cover mask and the tree outlines, the second calculates the distance transform of the masks (distance to next
        background pixel). The first net receives the input image, the second one receives the input image and the output of network 1.SK_SK_

        Args:
            in_channels: Number of input channels / bands of the input image
            architecture (str): segmentation model architecture
            backbone (str): segmentation model backbone
            lr: learning rate
            apply_sigmode (bool): TODO
            freeze_layers (bool): If True, freeze all layers but the segmentation head. Default: False.
            track_running_stats (bool): If True, update batch norm layers. If False, keep them frozen. Default: True.
            postprocessing_config (Dict[str, Any]): Set of parameters to apply in postprocessing (predict step). Default: {}.
        """
        super().__init__()
        self.seg_model = SegmentationModel(
            in_channels=in_channels,
            architecture=architecture,
            backbone=backbone,
            lr=lr,
            mask_loss_share=mask_iou_share,
        )
        self.dist_model = DistanceModel(
            in_channels=in_channels + 2, architecture=architecture, backbone=backbone
        )
        self.freeze_layers = freeze_layers
        self.track_running_stats = track_running_stats
        self.mask_iou_share = mask_iou_share
        self.postprocessing_config = postprocessing_config

        # freeze all layers but segmentation head
        if self.freeze_layers:
            for name, param in self.named_parameters():
                if "segmentation_head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # freeze the batch norm layers
        if not self.track_running_stats:
            for module in self.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
                    module.track_running_stats = False

        self.lr = lr
        self.apply_sigmoid = apply_sigmoid

    def forward(self, img):
        mask_and_outline = self.seg_model(img)
        dist = self.dist_model(img, mask_and_outline, from_logits=True)
        # dist = dist * mask_and_outline[:, [0]]
        if self.apply_sigmoid:
            return torch.cat((torch.sigmoid(mask_and_outline), dist), dim=1)
        else:
            return torch.cat((mask_and_outline, dist), dim=1)

    def shared_step(self, batch, iou_feature_map=False):
        """
        Perform a shared step in the model training or evaluation process.
        Args:
            batch (tuple): A tuple containing input data (x) and target data (y).
            iou_feature_map (bool, optional): If True, calculate the IoU for the feature map. Defaults to False.
        Returns:
            tuple: A tuple containing the following elements:
                - loss (torch.Tensor): The total loss combining mask, outline, and distance losses.
                - loss_mask (torch.Tensor): The loss for the mask prediction.
                - loss_outline (torch.Tensor): The loss for the outline prediction.
                - loss_distance (torch.Tensor): The loss for the distance prediction.
                - iou (torch.Tensor): The combined IoU of the mask and outline.
                - iou_mask (torch.Tensor): The IoU of the mask.
                - iou_outline (torch.Tensor): The IoU of the outline.
                - iou_fmap (torch.Tensor, optional): The IoU of the feature map, if `iou_feature_map` is True.
        """
        x, y = batch
        output = self(x)

        mask = output[:, 0]
        outline = output[:, 1]
        dist = output[:, 2]

        mask_t = y[:, 0]
        outline_t = y[:, 1]
        dist_t = y[:, 2]

        iou_mask = metrics.iou(torch.sigmoid(mask), mask_t)
        iou_outline = metrics.iou(torch.sigmoid(outline), outline_t)

        loss_mask = BinarySegmentationLossWithLogits(reduction="mean")(mask, mask_t)
        loss_outline = BinarySegmentationLossWithLogits(reduction="mean")(
            outline, outline_t
        )
        loss_distance = F.mse_loss(dist, dist_t)

        # lower mask loss results in unlearning the masks
        # lower distance loss results in artifacts in the distance transform
        # TODO add option to weight the different losses
        loss = loss_mask + loss_outline + loss_distance

        iou = self.mask_iou_share * iou_mask + (1.0 - self.mask_iou_share) * iou_outline

        if iou_feature_map:
            fmap_pred = tcdpp.calculate_feature_map(mask, outline, dist)
            fmap_target = tcdpp.calculate_feature_map(mask_t, outline_t, dist_t)
            iou_fmap = metrics.iou(fmap_pred, fmap_target)
            return (
                loss,
                loss_mask,
                loss_outline,
                loss_distance,
                iou,
                iou_mask,
                iou_outline,
                iou_fmap,
            )

        return loss, loss_mask, loss_outline, loss_distance, iou, iou_mask, iou_outline

    def training_step(self, batch, step):
        """
        Perform a single training step.
        Args:
            batch (Tensor): The input batch of data.
            step (int): The current training step.
        Returns:
            Tensor: The computed loss for the current training step.
        Logs:
            train/loss (float): The overall training loss.
            train/loss_mask (float): The training loss for the mask.
            train/loss_outline (float): The training loss for the outline.
            train/loss_dist (float): The training loss for the distance.
            train/iou (float): The Intersection over Union (IoU) metric.
            train/iou_mask (float): The IoU metric for the mask.
            train/iou_outline (float): The IoU metric for the outline.
        """
        loss, loss_mask, loss_outline, loss_distance, iou, iou_mask, iou_outline = (
            self.shared_step(batch)
        )
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
            "train/loss_dist",
            loss_distance,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("train/iou", iou, on_step=False, on_epoch=True, sync_dist=True)
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

    def validation_step(self, batch, step):
        """
        Performs a validation step during the training process.
        Args:
            batch (Tensor): The input batch of data.
            step (int): The current step in the validation process.
        Returns:
            Tensor: The computed loss for the validation step.
        Logs:
            val/loss (float): The overall validation loss.
            val/loss_mask (float): The validation loss for the mask.
            val/loss_outline (float): The validation loss for the outline.
            val/loss_dist (float): The validation loss for the distance.
            val/iou (float): The Intersection over Union (IoU) metric.
            val/iou_mask (float): The IoU metric for the mask.
            val/iou_outline (float): The IoU metric for the outline.
            val/iou_feature_map (float): The IoU metric for the feature map.
        """

        (
            loss,
            loss_mask,
            loss_outline,
            loss_distance,
            iou,
            iou_mask,
            iou_outline,
            iou_fmap,
        ) = self.shared_step(batch, iou_feature_map=True)
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
        self.log(
            "val/loss_dist", loss_distance, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log("val/iou", iou, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/iou_mask", iou_mask, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "val/iou_outline", iou_outline, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log(
            "val/iou_feature_map",
            iou_fmap,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, step):
        """
        Perform a test step on a given batch of data.
        Args:
            batch (Tensor): The input data batch.
            step (int): The current step number.
        Returns:
            dict: A dictionary containing the following keys:
            - 'mask': The mask output from the model.
            - 'outline': The outline output from the model.
            - 'distance_transform': The distance transform output from the model.
        """

        x = batch
        output = self(x)

        output_dict = {
            "mask": output[:, 0],
            "outline": output[:, 1],
            "distance_transform": output[:, 2],
        }

        return output_dict

    def predict_step(self, batch, step):
        """
        Perform a prediction step on a given batch of data.
        Args:
            batch (tuple): A tuple containing the input data and the transformation information.
            step (int): The current step number, used for saving prediction outputs.
        Returns:
            dict: A dictionary containing the extracted polygons and, if active learning is enabled,
                  the mean and median entropy values.
        This function performs the following steps:
        1. Runs inference on the input data to generate output predictions.
        2. Extracts mask, outline, and distance transform from the output.
        3. Saves the predictions if configured to do so.
        4. If active learning is enabled, calculates the probability map and entropy map, logs the mean and median entropy,
           and saves the entropy maps if configured to do so.
        5. Extracts polygons from the mask, outline, and distance transform using the specified postprocessing configuration.
        6. Logs the number of polygons found, inference time, and post-processing time.
        7. Returns a dictionary containing the extracted polygons and, if active learning is enabled, the mean and median entropy values.
        Note:
            - The function contains TODOs for using raster tile names and rasterizing outputs.
            - Additional postprocessing steps and trait extraction can be added as needed.
        """

        t0 = time.time()
        x, trafo = batch
        output = self(x)
        t_inference = time.time() - t0

        mask = output[:, 0, 6:-6, 6:-6].cpu().numpy().squeeze()
        outline = output[:, 1, 6:-6, 6:-6].cpu().numpy().squeeze()
        distance_transform = output[:, 2, 6:-6, 6:-6].cpu().numpy().squeeze()

        # TODO use the raster tile name here
        # TODO rasterize them (my gdal was not working ...)
        if self.postprocessing_config["save_predictions"]:
            np.save(f"./predictions/mask_{step}", mask)
            np.save(f"./predictions/outline_{step}", outline)
            np.save(f"./predictions/distance_transform_{step}", distance_transform)

        # active learning
        if self.postprocessing_config["active_learning"]:
            pmap = tcdpp.calculate_probability_map(
                mask,
                outline,
                distance_transform,
                mask_exp=self.postprocessing_config["mask_exp"],
                outline_multiplier=self.postprocessing_config["outline_multiplier"],
                outline_exp=self.postprocessing_config["outline_exp"],
                dist_exp=self.postprocessing_config["dist_exp"],
                sigma=self.postprocessing_config["sigma"],
            )
            entropy_map = tcdpp.calculate_entropy(pmap)
            log.info(f"Mean entropy: {np.mean(entropy_map):.4f}")
            log.info(f"Median entropy: {np.median(entropy_map):.4f}")

            # TODO use the raster tile name here
            # TODO rasterize them (my gdal was not working ...)
            if self.postprocessing_config["save_entropy_maps"]:
                np.save(f"./entropy_maps/entropy_heatmap_{step}", entropy_map)

        # add postprocessing here
        t0 = time.time()
        polygons = tcdpp.extract_polygons(
            mask,
            outline,
            distance_transform,
            transform=trafo,
            mask_exp=self.postprocessing_config["mask_exp"],
            outline_multiplier=self.postprocessing_config["outline_multiplier"],
            outline_exp=self.postprocessing_config["outline_exp"],
            dist_exp=self.postprocessing_config["dist_exp"],
            sigma=self.postprocessing_config["sigma"],
            binary_threshold=self.postprocessing_config["binary_threshold"],
            min_dist=self.postprocessing_config["min_dist"],
            label_threshold=self.postprocessing_config["label_threshold"],
            area_min=self.postprocessing_config["area_min"],
            simplify=self.postprocessing_config["simplify"],
        )

        t_process = time.time() - t0

        # TODO add option to extract the masked raster files of the polygons

        log.info(f"Found {len(polygons)} polygons.")
        log.info(f"Inference time: {t_inference:.2f} seconds")
        log.info(f"Post-processing time: {t_process:.2f} seconds")

        output_dict = {"polygons": polygons}

        if self.postprocessing_config["active_learning"]:
            output_dict["mean_entropy"] = np.mean(entropy_map)
            output_dict["median_entropy"] = np.median(entropy_map)

        # TODO add traits

        return output_dict

    def configure_optimizers(self):
        """
        Configures the optimizer for the model training.
        If `freeze_layers` is set to True, logs the names of the parameters that will be trained.
        Creates an Adam optimizer with the learning rate specified by `self.lr`, filtering out parameters
        that do not require gradients.
        Returns:
            list: A list containing the configured optimizer.
        """

        if self.freeze_layers:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    log.info(f"Parameters in {name} will be trained")
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, 2)
        # return [optimizer], [scheduler]
        return [optimizer]

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        architecture: str = "Unet",
        backbone: str = "resnet18",
        in_channels: int = 8,
    ):
        """
        Load a model from a checkpoint file.
        Args:
            path (str): The path to the checkpoint file.
            architecture (str, optional): The architecture of the segmentation model. Defaults to "Unet".
            backbone (str, optional): The backbone of the segmentation model. Defaults to "resnet18".
            in_channels (int, optional): The number of input channels for the segmentation model. Defaults to 8.
        Returns:
            An instance of the model loaded from the checkpoint.
        Raises:
            NotImplementedError: If the checkpoint cannot be loaded using the provided method.
        """
        seg_model = SegmentationModel(
            architecture=architecture, backbone=backbone, in_channels=in_channels
        )
        dist_model = DistanceModel(in_channels=in_channels + 2)
        try:
            return cls.load_from_checkpoint(
                path, segmentation_model=seg_model, distance_model=dist_model
            )
        except NotImplementedError:
            return torch.jit.load(path)

    def to_torchscript(self, method="trace", example_inputs=None):
        """
        Converts the model to TorchScript format using the specified method.
        Args:
            method (str): The method to use for converting the model to TorchScript.
                  Currently, only 'trace' is supported.
            example_inputs (tuple or torch.Tensor): Example inputs to be used for tracing the model.
                                Required if method is 'trace'.
        Returns:
            torch.jit.ScriptModule: The TorchScript representation of the model.
        Raises:
            ValueError: If an unsupported method is provided.
        """

        if method == "trace":
            return torch.jit.trace(self.forward, example_inputs=example_inputs)
        else:
            raise ValueError(method)
