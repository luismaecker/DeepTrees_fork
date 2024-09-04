'''
Fine tune a pretrained model on labeled Halle DOP data.

Follows the example script. 

Added:
- Early stopping callback monitoring the validation loss

TODO:
- Fix loading pretrained model checkpoint
- ColorJitter albumentation augmentation
- Fix model save

Caroline Arnold, Harsh Grover, Helmholtz AI, 2024
'''

import os

from treecrowndelineation import TreeCrownDelineationModel
from treecrowndelineation.dataloading.in_memory_datamodule import InMemoryDataModule

import albumentations as A

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="finetune")
def finetune(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    # we store the hyperparameters with the trained model and choose a short model name
    model_name = config['model']['model_name']

    # TODO store logs in different directory?
    logger = TensorBoardLogger(os.getcwd(), name=config['name'], version=model_name, default_hp_metric=False)

    callbacks = [
        #ModelCheckpoint(os.getcwd(), f'{model_name}-{epoch}', monitor='val/loss', save_last=True, save_top_k=2),
        ModelCheckpoint(os.path.join(os.getcwd(), 'checkpoints'), filename=None, monitor='val/loss', save_last=True, save_top_k=1),
        EarlyStopping(monitor='val/loss', patience=3, mode='min'),
        LearningRateMonitor()
    ]

    train_augmentation = A.Compose([A.RandomCrop(config['data']['width'], config['data']['width'], always_apply=True),
                                A.RandomRotate90(),
                                A.VerticalFlip(),
                                #A.ColorJitter(), TypeError: ColorJitter transformation expects 1-channel or 3-channel images.
                                ])
    val_augmentation = A.RandomCrop(config['data']['width'], config['data']['width'], always_apply=True)

    data = InMemoryDataModule(config['data']['rasters'],
                             (config['data']['masks'], config['data']['outlines'], config['data']['dist']),
                             width=config['data']['width'],
                             batchsize=config['data']['batchsize'],
                             training_split=config['data']['training_split'],
                             train_augmentation=train_augmentation,
                             val_augmentation=val_augmentation,
                             concatenate_ndvi=config['data']['concatenate_ndvi'],
                             red=config['data']['red'],
                             nir=config['data']['nir'],
                             dilate_second_target_band=2,
                             rescale_ndvi=True)

    model = TreeCrownDelineationModel(in_channels=config['model']['in_channels'], lr=config['model']['lr'])

    # FIXME figure this out
    #model = torch.jit.load('/work/ka1176/shared_data/2024-ufz-deeptree/pretrained_models/tcd-20cm-RGBI-v1/Unet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=0_jitted.pt')

    trainer = Trainer(devices=config['trainer']['devices'],
                      accelerator=config['trainer']['accelerator'],
                      logger=logger,
                      callbacks=callbacks,
                      max_epochs=config['trainer']['max_epochs']
                      )

    trainer.fit(model, data)

    # save the trained model
    # FIXME
    model.to('cpu')
    t = torch.rand(1, config['model']['in_channels'], config['data']['width'], config['data']['width'], dtype=torch.float32)
    model.to_torchscript(os.path.join(os.getcwd(), f'{model_name}_jitted.pt'), method='trace', example_inputs=t)

if __name__=='__main__':
    finetune()
