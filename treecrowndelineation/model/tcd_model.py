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

import torch
import lightning as L
from treecrowndelineation.model.segmentation_model import SegmentationModel
from treecrowndelineation.model.distance_model import DistanceModel
from treecrowndelineation.modules import metrics
from treecrowndelineation.modules.losses import BinarySegmentationLossWithLogits

import logging
log = logging.getLogger(__name__)

class TreeCrownDelineationModel(L.LightningModule):
    def __init__(self, in_channels,
                 architecture='Unet',
                 backbone='resnet18',
                 lr=1E-4,
                 mask_iou_share=0.5,
                 apply_sigmoid=False,
                 freeze_layers=False,
                 track_running_stats=True
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
        """
        super().__init__()
        self.seg_model = SegmentationModel(in_channels=in_channels, architecture=architecture, backbone=backbone, lr=lr, mask_loss_share=mask_iou_share)
        self.dist_model = DistanceModel(in_channels=in_channels + 2, architecture=architecture, backbone=backbone)
        self.freeze_layers = freeze_layers
        self.track_running_stats = track_running_stats
        self.mask_iou_share = mask_iou_share

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

    def shared_step(self, batch):
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
        loss_distance = torch.mean((dist - dist_t) ** 2)

        # lower mask loss results in unlearning the masks
        # lower distance loss results in artifacts in the distance transform
        # TODO add option to weight the different losses
        loss = loss_mask + loss_outline + loss_distance

        iou = self.mask_iou_share * iou_mask + (1.0 - self.mask_iou_share) * iou_outline

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
        loss, loss_mask, loss_outline, loss_distance, iou, iou_mask, iou_outline = self.shared_step(batch)
        self.log('val/loss'        , loss         , on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/loss_mask'   , loss_mask    , on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/loss_outline', loss_outline , on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/loss_dist'   , loss_distance, on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/iou'         , iou,           on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/iou_mask'    , iou_mask     , on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/iou_outline' , iou_outline  , on_step = False, on_epoch = True, sync_dist = True)
        return loss

    def configure_optimizers(self):
        if self.freeze_layers:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    log.info(f'Parameters in {name} will be trained')
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, 2)
        return [optimizer], [scheduler]

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
