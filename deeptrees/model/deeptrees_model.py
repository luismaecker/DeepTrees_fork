import rootutils

path = rootutils.find_root(search_from=__file__, indicator=".project-root")

# set root directory
rootutils.set_root(
    path=path, # path to the root directory
    project_root_env_var=True, # set the PROJECT_ROOT environment variable to root directory
    dotenv=True, # load environment variables from .env if exists in root directory
    pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
    cwd=False, # we do not want that with hydra
)

import time
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.functional.segmentation import mean_iou

from deeptrees.model.segmentation_model import SegmentationModel
from deeptrees.model.distance_model import DistanceModel
from deeptrees.modules import metrics
from deeptrees.modules.losses import BinarySegmentationLossWithLogits
from deeptrees.modules import postprocessing as tcdpp
from deeptrees.modules import utils

import logging
log = logging.getLogger(__name__)

class TreeCrownDelineationModel(L.LightningModule):
    def __init__(self, in_channels,
                 architecture='Unet',
                 backbone='resnet18',
                 lr=1E-4,
                 mask_iou_share=0.5,
                 apply_sigmoid=False,
                 ):
        """Tree crown delineation model

        The model consists of two sub-netoworks (two U-Nets with ResNet backbone). The first network calculates a tree
        cover mask and the tree outlines, the second calculates the distance transform of the masks (distance to next
        background pixel). The first net receives the input image, the second one receives the input image and the output of network 1.

        Args:
            in_channels: Number of input channels / bands of the input image
            architecture (str): segmentation model architecture
            backbone (str): segmentation model backbone
            lr: learning rate
            apply_sigmoid (bool): If True, apply sigmoid function to mask/outline outputs. Defaults to False.
        """
        super().__init__()
        self.seg_model = SegmentationModel(in_channels=in_channels, architecture=architecture, backbone=backbone, lr=lr, mask_loss_share=mask_iou_share)
        self.dist_model = DistanceModel(in_channels=in_channels + 2, architecture=architecture, backbone=backbone)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, img):
        mask_and_outline = self.seg_model(img)
        dist = self.dist_model(img, mask_and_outline, from_logits=True)
        # dist = dist * mask_and_outline[:, [0]]
        if self.apply_sigmoid:
            return torch.cat((torch.sigmoid(mask_and_outline), dist), dim=1)
        else:
            return torch.cat((mask_and_outline, dist), dim=1)
class DeepTreesModel(L.LightningModule):
    def __init__(self, in_channels,
                 architecture='Unet',
                 backbone='resnet18',
                 lr=1E-4,
                 mask_iou_share=0.5,
                 apply_sigmoid=False,
                 freeze_layers=False,
                 track_running_stats=True,
                 num_backbones=1,
                 postprocessing_config={}
                 ):
        """DeepTrees Model

        The model consists of a TreeCrownDelineation backbon, plus added functionalities for training, fine-tuning, and prediction.

        Args:
            in_channels: Number of input channels / bands of the input image
            architecture (str): segmentation model architecture
            backbone (str): segmentation model backbone
            lr: learning rate
            apply_sigmode (bool): TODO
            freeze_layers (bool): If True, freeze all layers but the segmentation head. Default: False.
            track_running_stats (bool): If True, update batch norm layers. If False, keep them frozen. Default: True.
            num_backbones (int): If > 1, instantiate several TCD backbones and average the model predictions across all of them. Defaults to 1.
            postprocessing_config (Dict[str, Any]): Set of parameters to apply in postprocessing (predict step). Default: {}.
        """
        super().__init__()
        self.num_backbones = num_backbones
        if self.num_backbones == 1:
            self.tcd_backbone = TreeCrownDelineationModel(in_channels=in_channels, architecture=architecture, backbone=backbone, lr=lr, mask_iou_share=mask_iou_share, apply_sigmoid=apply_sigmoid)
        else:
            tcd_dict = {}
            for i in range(self.num_backbones):
                tcd_dict[f'model_{i}'] = TreeCrownDelineationModel(in_channels=in_channels, architecture=architecture, backbone=backbone, lr=lr, mask_iou_share=mask_iou_share, apply_sigmoid=apply_sigmoid)
            self.tcd_backbone = nn.ModuleDict(tcd_dict)

        self.freeze_layers = freeze_layers
        self.track_running_stats = track_running_stats
        self.mask_iou_share = mask_iou_share
        self.lr = lr
        self.apply_sigmoid = apply_sigmoid
        self.postprocessing_config = postprocessing_config

        # freeze all layers but segmentation head
        if self.freeze_layers:
            for name, param in self.named_parameters():
                if 'segmentation_head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # freeze the batch norm layers
        if not self.track_running_stats:
            for module in self.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
                    module.track_running_stats = False

    def forward(self, img):
        if self.num_backbones == 1:
            return self.tcd_backbone(img)
        else:
            outputs = []
            for i in range(self.num_backbones):
                output = self.tcd_backbone[f'model_{i}'](img)
                outputs.append(output)
            
            outputs = torch.stack(outputs)
            return torch.mean(outputs, axis=0)

    def shared_step(self, batch, iou_feature_map=False):
        x, y = batch
        output = self(x)

        mask      = output[:, 0]
        outline   = output[:, 1]
        dist      = output[:, 2]

        mask_t    = y[:, 0]
        outline_t = y[:, 1]
        dist_t    = y[:, 2]

        iou_mask = metrics.iou(torch.sigmoid(mask), mask_t)
        iou_outline = metrics.iou(torch.sigmoid(outline), outline_t)

        loss_mask = BinarySegmentationLossWithLogits(reduction="mean")(mask, mask_t)
        loss_outline = BinarySegmentationLossWithLogits(reduction="mean")(outline, outline_t)
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
            return loss, loss_mask, loss_outline, loss_distance, iou, iou_mask, iou_outline, iou_fmap

        return loss, loss_mask, loss_outline, loss_distance, iou, iou_mask, iou_outline

    def training_step(self, batch, step):
        loss, loss_mask, loss_outline, loss_distance, iou, iou_mask, iou_outline = self.shared_step(batch)
        self.log('train/loss'        , loss         , on_step = False, on_epoch = True, sync_dist = True)
        self.log('train/loss_mask'   , loss_mask    , on_step = False, on_epoch = True, sync_dist = True)
        self.log('train/loss_outline', loss_outline , on_step = False, on_epoch = True, sync_dist = True)
        self.log('train/loss_dist'   , loss_distance, on_step = False, on_epoch = True, sync_dist = True)
        self.log('train/iou'         , iou,           on_step = False, on_epoch = True, sync_dist = True)
        self.log('train/iou_mask'    , iou_mask     , on_step = False, on_epoch = True, sync_dist = True)
        self.log('train/iou_outline' , iou_outline  , on_step = False, on_epoch = True, sync_dist = True)
        return loss

    def validation_step(self, batch, step):
        loss, loss_mask, loss_outline, loss_distance, iou, iou_mask, iou_outline, iou_fmap = self.shared_step(batch, iou_feature_map=True)
        self.log('val/loss'        , loss         , on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/loss_mask'   , loss_mask    , on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/loss_outline', loss_outline , on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/loss_dist'   , loss_distance, on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/iou'         , iou,           on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/iou_mask'    , iou_mask     , on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/iou_outline' , iou_outline  , on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/iou_feature_map', iou_fmap, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, step):
        x = batch
        output = self(x)

        output_dict = {'mask': output[:,0],
                       'outline': output[:,1],
                       'distance_transform': output[:,2]
                       }

        return output_dict

    def predict_step(self, batch, step):
        t0 = time.time()
        x, raster_dict = batch

        # predict step assumes batch size is 1 - extract first list entry
        assert x.shape[0] == 1, print("Predict batch size must be 1")
        trafo = raster_dict['trafo'][0]
        raster_name = raster_dict['raster_id'][0]
        raster_suffix = os.path.basename(raster_name).replace('tile_', '')
        output = self(x)
        t_inference = time.time() - t0

        mask = output[:,0,6:-6,6:-6].cpu().numpy().squeeze()
        outline = output[:,1,6:-6,6:-6].cpu().numpy().squeeze()
        distance_transform = output[:,2,6:-6,6:-6].cpu().numpy().squeeze()

        if self.postprocessing_config['save_predictions']:
            utils.array_to_tif(mask, f'./predictions/mask_{raster_suffix}', src_raster=raster_name)
            utils.array_to_tif(outline, f'./predictions/outline_{raster_suffix}', src_raster=raster_name)
            utils.array_to_tif(distance_transform, f'./predictions/distance_transform_{raster_suffix}', src_raster=raster_name)

        # active learning
        if self.postprocessing_config['active_learning']:
            pmap = tcdpp.calculate_probability_map(mask, outline, distance_transform, 
                                                mask_exp=self.postprocessing_config['mask_exp'],
                                                outline_multiplier=self.postprocessing_config['outline_multiplier'],
                                                outline_exp=self.postprocessing_config['outline_exp'],
                                                dist_exp=self.postprocessing_config['dist_exp'],
                                                sigma=self.postprocessing_config['sigma'])
            entropy_map = tcdpp.calculate_entropy(pmap)
            log.info(f'Mean entropy: {np.mean(entropy_map):.4f}')
            log.info(f'Median entropy: {np.median(entropy_map):.4f}')

            if self.postprocessing_config['save_entropy_maps']:
                utils.array_to_tif(entropy_map, f'./entropy_maps/entropy_heatmap_{raster_suffix}', src_raster=raster_name)

        # add postprocessing here
        t0 = time.time()
        polygons = tcdpp.extract_polygons(mask, outline, distance_transform, 
                                          transform=trafo,
                                          mask_exp=self.postprocessing_config['mask_exp'],
                                          outline_multiplier=self.postprocessing_config['outline_multiplier'],
                                          outline_exp=self.postprocessing_config['outline_exp'],
                                          dist_exp=self.postprocessing_config['dist_exp'],
                                          sigma=self.postprocessing_config['sigma'],
                                          binary_threshold=self.postprocessing_config['binary_threshold'],
                                          min_dist=self.postprocessing_config['min_dist'],
                                          label_threshold=self.postprocessing_config['label_threshold'],
                                          area_min=self.postprocessing_config['area_min'],
                                          simplify=self.postprocessing_config['simplify'])

        t_process = time.time() - t0

        # TODO add option to extract the masked raster files of the polygons

        log.info(f'Found {len(polygons)} polygons.')
        log.info(f'Inference time: {t_inference:.2f} seconds')
        log.info(f'Post-processing time: {t_process:.2f} seconds')
        
        output_dict = {'polygons': polygons}

        if self.postprocessing_config['active_learning']:
            output_dict['mean_entropy'] = np.mean(entropy_map)
            output_dict['median_entropy'] = np.median(entropy_map)

        # TODO add traits

        return output_dict

    def configure_optimizers(self):
        if self.freeze_layers:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    log.info(f'Parameters in {name} will be trained')
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, 2)
        #return [optimizer], [scheduler]
        return [optimizer]

    @classmethod
    def from_checkpoint(cls, path: str,
                        architecture: str = "Unet", backbone: str = "resnet18", in_channels: int = 8):
        seg_model = SegmentationModel(architecture=architecture,
                                      backbone=backbone,
                                      in_channels=in_channels)
        dist_model = DistanceModel(in_channels=in_channels + 2)
        try:
            return cls.load_from_checkpoint(path, segmentation_model=seg_model, distance_model=dist_model)
        except NotImplementedError:
            return torch.jit.load(path)

    def to_torchscript(self, method='trace', example_inputs=None):
        if method == 'trace':
            return torch.jit.trace(self.forward, example_inputs=example_inputs)
        else:
            raise ValueError(method)