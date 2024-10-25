'''
Train a TreeCrownDelineation model on new data.

Follows the example script. 

Added:
- Early stopping callback monitoring the validation loss

TODO:
- ColorJitter albumentation augmentation
- Does the validation dataloader need cropped images? Could work with the whole tile

Caroline Arnold, Harsh Grover, Helmholtz AI, 2024
'''

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


import os

from treecrowndelineation.model.tcd_model import TreeCrownDelineationModel
from treecrowndelineation.dataloading.in_memory_datamodule import InMemoryDataModule

import albumentations as A

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
log = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@hydra.main(version_base=None, config_path="../config", config_name="train_halle")
def train(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    if config.seed:
        seed_everything(config.seed, workers=True)

    # we store the hyperparameters with the trained model and choose a short model name
    model_name = config['model_name']

    # TODO store logs in different directory?
    logger = TensorBoardLogger(os.getcwd(), name=config['name'], version=model_name, default_hp_metric=False)

    callbacks = []
    for key, value in config.callbacks.items():
        if value is not None:
            log.info(f'Instantiating {key} callback')
            callbacks.append(hydra.utils.instantiate(value))
        else:
            log.info(f'Callback not instantiated: {key}')

    log.info('Instantiating data module ...')
    data: InMemoryDataModule = hydra.utils.instantiate(config.data)

    log.info('Instantiating model...')
    model: TreeCrownDelineationModel = hydra.utils.instantiate(config.model)

    if config['pretrained']['model'] is not None:
        pretrained_model = torch.jit.load(os.path.join(config['pretrained']['path'], config['pretrained']['model']))
        model.load_state_dict(pretrained_model.state_dict())
        log.info('Loaded state dict from pretrained model')

    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    log.info('Starting model fit')
    trainer.fit(model, data)

    # save the trained model
    log.info('Saving trained model')
    model.to('cpu')
    input_sample = torch.rand(1, config['model']['in_channels'], config['data']['width'], config['data']['width'], dtype=torch.float32)
    torch.jit.save( model.to_torchscript(method='trace', example_inputs=input_sample), os.path.join(os.getcwd(), f'{model_name}_jitted.pt') )
    log.info(f'Saved torchscript to {os.getcwd():s}/{model_name:s}_jitted.pt')
    log.info('Completed!')

if __name__=='__main__':
    train()
