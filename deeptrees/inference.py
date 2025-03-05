"""
Tree Crown Delineation Inference Script

This script performs tree crown delineation using a pre-trained DeepTreesModel. It loads the model configuration,
initializes the model, and runs inference on input raster images to predict tree crowns. The predictions are saved
as raster files, and post-processing is performed to extract polygons representing tree crowns.

Classes:
    TreeCrownPredictor: A class to handle the loading of the model, running inference, and post-processing.

Usage:
    python inference.py

Example:
    predictor = TreeCrownPredictor(config_path="./config", image_path=["/path/to/raster/image.tif"])
    predictor.predict('/path/to/raster/image.tif', '/path/to/config')
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
from omegaconf import OmegaConf

from .pretrained import freudenberg2022
from .model.deeptrees_model import DeepTreesModel
from .modules import utils
from .dataloading.datasets import TreeCrownDelineationInferenceDataset
import time
from .modules import postprocessing as tcdpp
from .modules.utils import mask_and_save_individual_trees


logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',  # Set the date format to exclude milliseconds
    handlers=[
        logging.StreamHandler(sys.stdout)  # Output to stdout
    ]
)
log = logging.getLogger(__name__)



class TreeCrownPredictor:
    """
    A class to handle the loading of the model, running inference, and post-processing.

    Attributes:
        config (OmegaConf): The configuration loaded from a YAML file.
        model (DeepTreesModel): The deep learning model for tree crown delineation.
        image_path (str): The path to the input raster image.
        dataset (TreeCrownDelineationInferenceDataset): The dataset for inference.

    Methods:
        _initialize_model(): Initializes the model with the configuration parameters.
        predict(): Runs inference on the input data and performs post-processing and saves the results.
    """

    def __init__(self, image_path: list[str] = None, config_path: str = "./config/inference_on_individual_tiles.yaml"):
        """
        Initializes the TreeCrownPredictor with the given configuration.

        Args:
            config_path (str): The path to the configuration folder.            
            image_path (str): The path to the input raster image.
        """
        # Load the config using OmegaConf
        self.config = OmegaConf.load(os.path.join(config_path))
        
        # Print the loaded configuration to the console
        log.info("Loaded Configuration:")
        log.info(OmegaConf.to_yaml(self.config))  # This prints the entire configuration in YAML format
        
        self.model = None
        
        # Check if the image path is provided and is valid list of strings        
        if image_path and isinstance(image_path, list):
            self.image_path = image_path
        else:
            raise ValueError("Please provide input raster image/s path as a list")
        
        self.dataset = TreeCrownDelineationInferenceDataset(
                raster_files=self.image_path,
                augmentation=self.config.data.augment_eval,
                ndvi_config=self.config.data.ndvi_config,
                dilate_outlines=self.config.data.dilate_outlines,
                in_memory=False,
                divide_by=self.config.data.divide_by)
    
        self._initialize_model()      
        
        # Directories for saving output
        if self.config['save_predictions']:
            if not os.path.exists('predictions'):
                os.mkdir('predictions')
        if self.config['save_entropy_maps']:
            if not os.path.exists('entropy_maps'):
                os.mkdir('entropy_maps')
  

    def _initialize_model(self):
        """
        Initializes the model with the configuration parameters.
        """

        pretrained_model_file = os.path.join(self.config['pretrained_model_path'], self.config['pretrained_model_name'])

        # Access the model parameters from config and pass them to the constructor manually
        model_config = self.config.model
   
        self.model = DeepTreesModel(
                num_backbones=model_config.num_backbones,
                in_channels = model_config.in_channels,
                architecture= model_config.architecture,
                backbone=model_config.backbone,                                  
                apply_sigmoid=model_config.apply_sigmoid,
        )

        if self.config['download_pretrained_model']:
            if not os.path.exists(pretrained_model_file):
                os.makedirs('./pretrained_models', exist_ok=True)
                freudenberg2022(pretrained_model_file)        
        
        # check this after the model download
        if not isinstance(pretrained_model_file, str):
            raise ValueError("Pretrained model file path must be passed.")

        log.info(f'Loading state dict from pretrained model at :{pretrained_model_file}')
        pretrained_model = torch.jit.load(pretrained_model_file)
        self.model.tcd_backbone.load_state_dict(pretrained_model.state_dict())
        log.info(f'Finished loading state dict')
        

    def predict(self):
        """
        Runs inference on the input data and performs post-processing.
        """
        # Run inference on the data
        self.model.eval()  # Set the model to evaluation mode        

        for idx, batch in enumerate(self.dataset):
            t0 = time.time()
            raster, raster_dict = batch            
            trafo = raster_dict['trafo'] 
            raster_name = raster_dict['raster_id']
            raster_suffix = os.path.basename(raster_name).replace('tile_', '')
            log.info(f"Predicting on {raster_name} ...")
            
            raster = raster.unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                output = utils.predict_on_tile(self.model, raster)
                
            t_inference = time.time() - t0

            mask = output[:,0].cpu().numpy().squeeze()
            outline = output[:,1].cpu().numpy().squeeze()
            distance_transform = output[:,2].cpu().numpy().squeeze()

            if self.config['save_predictions']:
                utils.array_to_tif(mask, f'./predictions/mask_{raster_suffix}', src_raster=raster_name, num_bands='single')                
                utils.array_to_tif(outline, f'./predictions/outline_{raster_suffix}', src_raster=raster_name, num_bands='single')
                utils.array_to_tif(distance_transform, f'./predictions/distance_transform_{raster_suffix}', src_raster=raster_name, num_bands='single')
                log.info(f"Saved Mask, Outline and Distance Transform output to {os.path.join(os.getcwd(), 'predictions')}")

            # active learning
            if self.config["active_learning"]:
                pmap = tcdpp.calculate_probability_map(
                    mask,
                    outline,
                    distance_transform,
                    mask_exp=self.config.polygon_extraction["mask_exp"],
                    outline_multiplier=self.config.polygon_extraction["outline_multiplier"],
                    outline_exp=self.config.polygon_extraction["outline_exp"],
                    dist_exp=self.config.polygon_extraction["dist_exp"],
                    sigma=self.config.polygon_extraction["sigma"]
                )
                
                entropy_map = tcdpp.calculate_entropy(pmap)
                log.info(f"Mean entropy in {os.path.basename(raster_name)}: {np.mean(entropy_map):.4f}")
                log.info(f"Max entropy in {os.path.basename(raster_name)}: {np.max(entropy_map):.4f}")
                
                if self.config['save_entropy_maps']:
                    utils.array_to_tif(entropy_map, f'./entropy_maps/entropy_heatmap_{raster_suffix}', src_raster=raster_name)
                    
                log.info('Saving entropy map to ./entropy_maps')
                    
            # Extract polygons 
            t0 = time.time()
            polygons = tcdpp.extract_polygons(
                mask,
                outline,
                distance_transform,
                transform=trafo,
                mask_exp=self.config.polygon_extraction["mask_exp"],
                outline_multiplier=self.config.polygon_extraction["outline_multiplier"],
                outline_exp=self.config.polygon_extraction["outline_exp"],
                dist_exp=self.config.polygon_extraction["dist_exp"],
                sigma=self.config.polygon_extraction["sigma"],
                binary_threshold=self.config.polygon_extraction["binary_threshold"],
                min_dist=self.config.polygon_extraction["min_dist"],
                label_threshold=self.config.polygon_extraction["label_threshold"],
                area_min=self.config.polygon_extraction["area_min"],
                simplify=self.config.polygon_extraction["simplify"]
            )
            
            t_process = time.time() - t0
                
            log.info(f"Found {len(polygons)} polygons.")
            log.info(f"Inference time: {t_inference:.2f} seconds")
            log.info(f"Post-processing time: {t_process:.2f} seconds")
                
            # create a diectory and save all above polygons to that directory
            polygon_file =  os.path.basename(raster_name).split('.')[0] + '.shp'
            os.makedirs('./saved_polygons', exist_ok=True)
            polygon_path = os.path.join(os.getcwd(), 'saved_polygons' ,polygon_file)
            log.info(f'Saving all polygons to {polygon_path}')
            utils.save_polygons(polygons, polygon_path, crs=self.config['crs'])
            
            
            # save individual trees as rasters
            if self.config.save_masked_rasters:
            
                log.info(f"Saving mask_and_scale_raster_from_polygons to {self.config.masked_rasters_output_dir}.")
                
                mask_and_save_individual_trees(tiff_path=raster_name,
                                                    polygons=polygons,                                                         
                                                    output_dir=os.path.join(self.config.masked_rasters_output_dir, 
                                                                            raster_suffix.split('.')[0]),
                                                                            scale_factor=self.config.scale_factor)
        

# For command parsing run the main function
def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Predict tree crown from image tiles.")
    
    parser.add_argument(
        "--image_path", 
        default="",
        nargs='+',  # Accept multiple paths
        help="List of image paths to process.", 
        required=True        
    )
    
        
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,    
        default="",   
        help="Directory containing the configuration file."
    )
    
    
    # Parse the arguments
    args = parser.parse_args()


    # Check if the image path is provided and is valid list of strings        
    if not isinstance(args.image_path, list):
        args.image_path = [args.image_path]
        
    # Initialize the predictor with the passed with configuration and image paths
    predictor = TreeCrownPredictor(image_path=args.image_path, config_path=args.config_path)


    # Run the prediction
    predictor.predict()


if __name__ == "__main__":
    main()
