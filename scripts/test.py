'''
Inference with a trained TreeCrownDelineation model

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
from lightning.pytorch.loggers import MLFlowLogger

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
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

from deeptrees.model.tcd_model import TreeCrownDelineationModel
from deeptrees.model.averaging_model import AveragingModel
from deeptrees.dataloading.datamodule import TreeCrownDelineationDataModule
from deeptrees.modules import utils

import geopandas as gpd

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="inference_halle")
def test(config: DictConfig) -> None:
    '''
    Inference with a pretrained tree crown delineation model.

    Args:
        config (DictConfig): configuration (provided by hydra)
    '''
    print(OmegaConf.to_yaml(config))

    if config.seed:
        seed_everything(config.seed, workers=True)

    callbacks = []
    for key, value in config.callbacks.items():
        if value is not None:
            log.info(f'Instantiating {key} callback')
            callbacks.append(hydra.utils.instantiate(value))
        else:
            log.info(f'Callback not instantiated: {key}')

    log.info('Instantiating data module ...')
    data: TreeCrownDelineationDataModule = hydra.utils.instantiate(config.data)
    data.prepare_data()
    data.setup(stage='test')

    log.info('Instantiating model...')
    model: TreeCrownDelineationModel = hydra.utils.instantiate(config.model)

    if isinstance(config['pretrained_model'], str):
        pretrained_model = torch.jit.load(config['pretrained_model'])
        model.load_state_dict(pretrained_model.state_dict())
        log.info('Loaded state dict from pretrained model')
    elif isinstance(config['pretrained_model'], ListConfig) or isinstance(config['pretrained_model'], list):
        pretrained_models = [torch.jit.load(m) for m in config['pretrained_model']]
        averaged_state_dict = {}
        for key, val in model.state_dict().items():
            if key.endswith('num_batches_tracked'):
                continue
            mval = [pm.state_dict()[key] for pm in pretrained_models]
            mval = torch.stack(mval, axis=0)
            mval = torch.mean(mval, axis=0)
            averaged_state_dict[key] = mval.to(val.dtype)

        model.load_state_dict(averaged_state_dict)
        log.info(f'Loaded state dict from pretrained models, averaging the weights of {len(config["pretrained_model"])} models.')
    else:
        raise ValueError("Inference requires a pretrained model")

    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks)

    # Directories for saving output
    if config['model']['postprocessing_config']['save_predictions']:
        if not os.path.exists('predictions'):
            os.mkdir('predictions')
    if config['model']['postprocessing_config']['save_entropy_maps']:
        if not os.path.exists('entropy_maps'):
            os.mkdir('entropy_maps')

    log.info('Starting predictions ...')
    output_dict = trainer.predict(model, data)

    all_polygons = []
    for dd in output_dict:
        all_polygons.extend(dd['polygons'])

    log.info(f'Saving all polygons to {os.path.join(os.getcwd(), config["polygon_file"])}.')
    utils.save_polygons(all_polygons, config['polygon_file'], crs=config['crs'])

    # additional post processing that works on all polygons
    baumkataster = gpd.read_file(config['baumkataster_file']).to_crs(config['crs'])
    inters = gpd.GeoDataFrame(geometry=all_polygons).set_crs(config['crs']).sjoin(baumkataster)
    if len(inters) == 0:
        log.info(f'No polygons found that overlap with Baumkataster.')
    else:
        log.info(f'Saving all polygons that overlap with Baumkataster to {os.path.join(os.getcwd(), config["baumkataster_intersection_file"])}.')
        utils.save_polygons(inters['geometry'], config['baumkataster_intersection_file'], crs=config['crs'])


if __name__=='__main__':
    test()
