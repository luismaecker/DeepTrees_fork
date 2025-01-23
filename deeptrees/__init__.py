__author__ = "Max Freudenberg"
__version__ = "0.1.0"

from . import model
from . import modules
from . import dataloading

from .model.deeptrees_model import TreeCrownDelineationModel

from deeptrees.inference import TreeCrownPredictor

def predict(image_path: list[str]):
    """
    Run tree crown delineation prediction on the provided image path.

    Args:
        image_path (list[str]): List of image paths to process.
    """
    predictor = TreeCrownPredictor(image_path=image_path)  # Uses default config path and name
    predictor.predict()