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