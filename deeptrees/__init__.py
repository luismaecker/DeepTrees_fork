__version__ = "v1.6.0"

from . import model
from . import modules
from . import dataloading

from .model.deeptrees_model import TreeCrownDelineationModel

from .inference import TreeCrownPredictor

from .pretrained import freudenberg2022

import os


def predict(image_path: list[str], config_path: str):
    """
    Run tree crown delineation prediction on the provided image paths using the given configuration.

    Args:
        image_path (list[str]): A list of file paths to the images to be processed.
        config_path (str): The file path to the configuration file for the prediction.

    Returns:
        None: This function does not return any value. It performs the prediction in-place.
    """
    predictor = TreeCrownPredictor(image_path=image_path, config_path=config_path)  # Uses default config path and name
    predictor.predict()


def download_pre_trained_model_weights():
    """
    Downloads the pretrained model weights for the Freudenberg et al. 2022 model.
    
    Args:
        download_path (str): The path to save the downloaded model weights to.
    """
    
    os.makedirs('./pretrained_models', exist_ok=True)
    
    file_name = os.path.join('./pretrained_models', "lUnet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=3_jitted.pt")
    
    freudenberg2022(file_name)