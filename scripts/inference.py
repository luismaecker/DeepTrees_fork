'''
TODO update module docstring
Train a TreeCrownDelineation model on new data.

Follows the example script. 

Caroline Arnold, Harsh Grover, Helmholtz AI, 2024

===================================================================================================
References:

M. Freudenberg, P. Magdon, and N. Nölke. Individual tree crown delineation in high-resolution 
remote sensing images based on U-Net.  NCAA (2022). https://doi.org/10.1007/s00521-022-07640-4

https://github.com/AWF-GAUG/TreeCrownDelineation (v0.1.0). (c) Max Freudenberg, MIT License.
===================================================================================================
'''

import warnings
import logging
import os

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

import hydra
from omegaconf import DictConfig, OmegaConf
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

from treecrowndelineation.model.tcd_model import TreeCrownDelineationModel
from treecrowndelineation.dataloading.datamodule import TreeCrownDelineationDataModule

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="inference_halle")
def test(config: DictConfig) -> None:
    '''
    Inference with a TreeCrownDelineation segmentation model

    Configuration options include
    - active learning
    - database of extracted polygons
    - overlap of extracted polygons with Baumkataster tree locations

    Args:
        config (DictConfig): configuration (provided by hydra)
    '''
    print(OmegaConf.to_yaml(config))

    if config.seed:
        seed_everything(config.seed, workers=True)

    # we store the hyperparameters with the trained model and choose a short model name
    model_name = config['model_name']

    for key, value in config.callbacks.items():
        if value is not None:
            log.info(f'Instantiating {key} callback')
            callbacks.append(hydra.utils.instantiate(value))
        else:
            log.info(f'Callback not instantiated: {key}')

    log.info('Instantiating data module ...')
    data: TreeCrownDelineationDataModule = hydra.utils.instantiate(config.data)
    data.prepare_data()
    data.setup()

    log.info('Instantiating model...')
    model: TreeCrownDelineationModel = hydra.utils.instantiate(config.model)

    if config['pretrained']['model'] is not None:
        pretrained_model = torch.jit.load(os.path.join(config['pretrained']['path'],
                                                       config['pretrained']['model']))
        model.load_state_dict(pretrained_model.state_dict())
        log.info('Loaded state dict from pretrained model')

    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    log.info('Starting model fit')
    trainer.fit(model, data)

    # save the trained model
    log.info('Saving trained model')
    model.to('cpu')
    input_sample = torch.rand(1,
                              config['model']['in_channels'],
                              config['data']['width'],
                              config['data']['width'],
                              dtype=torch.float32)
    torch.jit.save(model.to_torchscript(method='trace', example_inputs=input_sample),
                   os.path.join(os.getcwd(), f'{model_name}_jitted.pt'))
    log.info(f'Saved torchscript to {os.getcwd():s}/{model_name:s}_jitted.pt')
    log.info('Completed!')

if __name__=='__main__':
    test()
