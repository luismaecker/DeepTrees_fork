'''
Inference with a trained model on digital orthophotos (DOP) from Halle.

Follows the script inference.py. 

Added:

TODO:
- load more than one model and average them
- Enable inference on a folder of tifs - Set this up with a proper dataloader

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
import fiona
import torch
import psutil
import xarray
import numpy as np
from torch.nn import DataParallel
from torch.nn import UpsamplingBilinear2d, Sequential
import time
from treecrowndelineation.modules import utils
from treecrowndelineation.modules.indices import ndvi
from treecrowndelineation.modules.postprocessing import extract_polygons
from treecrowndelineation.modules.utils import get_crs
from treecrowndelineation.model.inference_model import InferenceModel
from treecrowndelineation.model.averaging_model import AveragingModel

from baal import ModelWrapper
from baal.active import get_heuristic, ActiveLearningLoop
from baal.bayesian.dropout import MCDropoutModule
from baal.modelwrapper import TrainingArgs
from baal.utils.metrics import ClassificationReport
from baal.utils.transforms import PILToLongTensor


import hydra
from omegaconf import DictConfig, OmegaConf

import logging
log = logging.getLogger(__name__)

def calculate_entropy(probability_map):
    # Ensure the probability map is clipped between a very small value to avoid log(0)
    probability_map = np.clip(probability_map, 1e-6, 1 - 1e-6)
    entropy = - (probability_map * np.log(probability_map) + (1 - probability_map) * np.log(1 - probability_map))
    return entropy

@hydra.main(version_base=None, config_path="../config", config_name="inference_halle")
def test(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    polygon_extraction_params = {"min_dist"          : config.min_dist,
                                 "mask_exp"          : 2,
                                 "outline_multiplier": 5,
                                 "dist_exp"          : 0.5,
                                 "sigma"             : config.sigma,
                                 "label_threshold"   : config.label_threshold,
                                 "binary_threshold"  : config.binary_threshold,
                                 "simplify"          : config.simplify_dist,
                                 }

    log.info("Loading model")
    
    if isinstance(config.model_path, str):
        model = torch.jit.load(config.model_path).to(config.device)

    elif isinstance(config.model_path, list):
        models = [torch.jit.load(m).to(config.device) for m in config.model_path]
        model = AveragingModel(models)
    else:
        raise RuntimeError('Model loading failed')

    log.info("Model loaded")

    stride = config.stride or config.width - 32 - config.width // 10

    if config.upsample != 1:
        model = Sequential(UpsamplingBilinear2d(scale_factor=config.upsample), model, UpsamplingBilinear2d(
                scale_factor=1. / config.upsample))

    if config.sigmoid:
        model = InferenceModel(model)  # apply sigmoid to mask and outlines, but not to distance transform

    model = DataParallel(model) # TODO check if we need this - should it be handled by the trainer?
    model.eval()
    # TODO: Add this path to hydra config file
    ground_truth_path = '/work/ka1176/shared_data/Deeptree_data_04_Oct_2024/polygon-labelling/tiles/'
    chunk_means = [] 
    for idx, img in enumerate(os.listdir(ground_truth_path)):
        filename = os.path.basename(img).split('.')[0]
        
        array = xarray.open_rasterio(os.path.join(ground_truth_path, img))
        nbands, height, width = array.shape

        if config.tilesize == 0:
            free_mem_bytes = psutil.virtual_memory().available
            byes_per_pixel = nbands * 4  # float32
            total_pixels = free_mem_bytes / byes_per_pixel / 8
            chunk_size = min(max(width, height), int(np.sqrt(total_pixels)))
        else:
            chunk_size = config.tilesize

        nchunks_h = len(range(0, height, chunk_size))
        nchunks_w = len(range(0, width, chunk_size))
        
        print('total chunks in heights and width',nchunks_h, nchunks_w)
        nchunks = nchunks_h * nchunks_w
        log.info("Chunk size for processing: {} pixels".format(chunk_size))

        log.info("Starting processing...")

        polygons = []

        inference_time = 0
        postprocessing_time = 0
        disk_loading_time = 0


        t0 = time.time()
        
        for i, y in enumerate(range(0, height, chunk_size)):             
            for j, x in enumerate(range(0, width, chunk_size)):
                idx = i * nchunks_w + j + 1
                log.info("Loading chunk {}/{}".format(idx, nchunks))
                t1 = time.time()
                chunk = array[:, y:y + chunk_size, x:x + chunk_size].load()  #.transpose('y', 'x', 'band')
                data = chunk.data
                if config.divisor != 1:
                    data = data / config.divisor

                if config.ndvi:
                    # data = np.concatenate((data, ndvi(data, red=config.red, nir=config.nir, axis=2)[...,None]), axis=2)
                    n = ndvi(data, red=config.red, nir=config.nir, axis=0)[None, ...]
                    if config.rescale_ndvi:
                        n = (n + 1) / 2
                    data = np.concatenate((data, n), axis=0)

                t2 = time.time()
                disk_loading_time += t2 - t1
                log.info("Starting prediction on chunk {}/{}".format(idx, nchunks))
                result = utils.predict_on_array_cf(model,
                                                    data,
                                                    in_shape=(nbands + config.ndvi, config.width,
                                                    config.width),
                                                    out_bands=3,
                                                    drop_border=16,
                                                    batchsize=config.batchsize,
                                                    stride=stride,
                                                    device=config.device,
                                                    augmentation=config.augment,
                                                    no_data=0,
                                                    verbose=True)
                
                # Extract prediction array from the result dictionary
                prediction_array = result['prediction']

                # Extract only the first channel, which represents the probability map
                probability_channel_2 = prediction_array[2, :, :]


                # Assuming `probability_channel_2` is the probability map of the foreground
                entropy_map = calculate_entropy(probability_channel_2)
                mean_entropy = np.mean(entropy_map)

                # You can then sort patches by mean_entropy to select the least certain ones
                chunk_means.append((mean_entropy, filename))

                log.info(f"Mean entropy of chunk {filename}: {mean_entropy}")


                t3 = time.time()
                inference_time += t3 - t2

                (ymin, ymax, xmin, xmax) = result["nodata_region"]

                if result["prediction"] is None:
                    # in this case the prediction area was all no data, nothing to extract from here
                    log.info("Empty chunk, skipping remaining steps.")
                    continue

                log.info("Prediction done, extracting polygons for chunk {}/{}.".format(idx, nchunks))

                if config.save_prediction is not None:
                    utils.array_to_tif(result["prediction"].transpose(1, 2, 0),
                                        os.path.abspath(config.save_prediction) + '/' + filename + "_pred.tif",
                                        transform=utils.xarray_trafo_to_gdal_trafo(chunk.attrs["transform"]),
                                        crs=array.attrs["crs"])

                t4 = time.time()

                if config.subsample:
                    xres, xskew, xr, yskew, yres, yr = utils.get_xarray_trafo(chunk[:, ymin:ymax, xmin:xmax])
                    xres *= 2
                    yres *= 2
                    trafo = (xres, xskew, xr, yskew, yres, yr)
                    config.sigma /= 2
                    config.min_dist /= 2

                    polygons.extend(extract_polygons(*result["prediction"][:, ymin:ymax:2, xmin:xmax:2],
                                                        transform=trafo,
                                                        area_min=3,
                                                        **polygon_extraction_params))

                else:
                    trafo = utils.get_xarray_trafo(chunk[:, ymin:ymax, xmin:xmax])
                    polygons.extend(extract_polygons(*result["prediction"][:, ymin:ymax, xmin:xmax],
                                                        transform=trafo,
                                                        area_min=3,
                                                        **polygon_extraction_params))
                t5 = time.time()
                postprocessing_time += t5 - t4
        
    # Sort the chunk means in ascending order by the mean probability value
    sorted_chunk_means = sorted(chunk_means, key=lambda x: x[0], reverse=True)

    # Print or log the sorted list of chunk means
    for mean, chunk_id in sorted_chunk_means:
        log.info(f"Chunk {chunk_id} has mean entropy: {mean}")
        
    log.info("Found {} polygons in total.".format(len(polygons)))
    log.info("Total processing time: {}s".format(int(time.time() - t0)))
    log.info("Time loading from disk: {}s".format(int(disk_loading_time)))
    log.info("Inference time: {}s".format(int(inference_time)))
    log.info("Post-processing time: {}s".format(int(postprocessing_time)))
    log.info("Saving as {}".format(os.path.join(os.getcwd(), config.output_file)))

    crs_ = get_crs(array)

    utils.save_polygons(polygons,
                        os.path.join(os.getcwd(), config.output_file),
                        crs=crs_)
    log.info("Done.")

if __name__ == '__main__':
    test()
