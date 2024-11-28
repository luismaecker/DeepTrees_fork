from typing import Union, Dict, Any

import os
import glob
import numpy as np
import lightning as L
import pandas as pd
import geopandas as gpd

from torch.utils.data import DataLoader
from deeptrees.dataloading import datasets as ds
from deeptrees.modules.utils import dilate_img, fix_crs
from deeptrees.dataloading.preprocessing import (
    MaskOutlinesGenerator,
    DistanceTransformGenerator,
)

import logging

log = logging.getLogger(__name__)


class TreeCrownDelineationDataModule(L.LightningDataModule):
    def __init__(
        self,
        rasters: Union[str, list],
        masks: Union[str, list],
        outlines: Union[str, list],
        distance_transforms: Union[str, list],
        training_split: float = 0.7,
        batch_size: int = 16,
        val_batch_size: int = 2,
        num_workers: int = 8,
        augment_train: Dict[str, Any] = {},
        augment_eval: Dict[str, Any] = {},
        ndvi_config: Dict[str, Any] = {"concatenate": False},
        divide_by: float = 1,
        dilate_outlines: int = 0,
        shuffle: bool = True,
        train_indices: list[int] = None,
        val_indices: list[int] = None,
        test_indices: list[int] = None,
        ground_truth_config: Dict[str, Any] = {"labels": None},
    ):
        """
        TreeCrownDelineationDataModule

        Datamodule to hold the different datasets, apply preprocessing, and return DataLoaders.

        Args:
            rasters (Union[str, list]): List of file paths, or path to folder containing the training raster files (TIF).
            masks (Union[str, list]): List of file paths, or path to folder containing the masks.
            outlines (Union[str, list]): List of file paths, or path to folder containing the outlines.
            distance_transforms (Union[str, list]): List of file paths, or path to folder containing the distance transforms.
            training_split (float, optional): Training data split. Defaults to 0.7.
            batch_size (int, optional): Training batch size. Defaults to 16.
            val_batch_size (int, optional): Validation batch size. Defaults to 2.
            num_workers (int, optional): Number of workers in DataLoader. Defaults to 8.
            augment_train (Dict[str, Any], optional): Dictionary defining torchvision augmentations to be used during training. Defaults to {}.
            augment_eval (Dict[str, Any], optional): Dictionary defining torchvision augmentations to be used during validation/testing. Defaults to {}.
            ndvi_config (_type_, optional): Dictionary defining the NDVI concatenation settings. Defaults to {'concatenate': False}.
            divide_by (float, optional): Scalar used to normalize rasters. Defaults to 1.
            dilate_outlines (int, optional): If present (>0), dilate outlines be given number of pixels. Defaults to False (=0).
            shuffle (bool, optional): If True, shuffle data before applying split. Defaults to True.
            train_indices (list[int], optional): List of indices of files to be used for training. Cannot be used with shuffle. Defaults to None.
            val_indices (list[int], optional): List of indices of files to be used for validation. Cannot be used with shuffle. Defaults to None.
            test_indices (list[int], optional): List of indices of files to be used for testing. Cannot be used with shuffle. Defaults to None.
            ground_truth_config (Dict[str, Any], optional): Dictionary defining the ground truth preprocessing settings. Defaults to {'labels': None}.
        """
        super().__init__()
        if type(rasters) in (list, tuple, np.ndarray):
            self.rasters = rasters
        else:
            self.rasters = np.sort(glob.glob(os.path.abspath(rasters) + "/*.tif"))

        self.masks = masks
        self.outlines = outlines
        self.distance_transforms = distance_transforms

        self.training_split = training_split
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.augment_train = augment_train
        self.augment_eval = augment_eval
        self.ndvi_config = ndvi_config
        self.ground_truth_config = ground_truth_config
        self.dilate_outlines = dilate_outlines
        self.shuffle = shuffle
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.divide_by = divide_by
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.targets = None  # will be assigned in setup_data

    def prepare_data(self) -> None:
        """prepare_data

        Prepare the ground truth masks, outlines, and distance transforms from
        ground truth labels.
        """

        if self.ground_truth_config.labels is None:
            log.info(
                "No ground truth labels provided. Proceed with existing ground truth ..."
            )
            log.info(f"Masks: {self.masks}")
            log.info(f"Outlines: {self.outlines}")
            log.info(f"Distance transforms: {self.distance_transforms}")

            return

        # prepare ground truth from labels
        if os.path.isfile(self.ground_truth_config.labels):
            ground_truth = gpd.read_file(self.ground_truth_config.labels)
        elif os.path.isdir(self.ground_truth_config.labels):
            # combine all the ground truth labels
            shapes = np.sort(
                glob.glob(f"{self.ground_truth_config.labels}/label_*.shp")
            )
            ground_truth = pd.concat(
                [fix_crs(gpd.read_file(shape)).assign(tile=shape) for shape in shapes]
            )
            log.info(
                f'Combining all polygons in {os.path.join(self.ground_truth_config.labels, "all_labels.shp")}'
            )
            ground_truth.drop(columns="tile").to_file(
                os.path.join(self.ground_truth_config.labels, "all_labels.shp")
            )

        # generate masks
        mask_generator = MaskOutlinesGenerator(
            rasters=self.rasters,
            output_path=self.masks,
            output_file_prefix="mask",
            ground_truth_labels=ground_truth,
            valid_class_ids=self.ground_truth_config.valid_class_ids,
            class_column_name=self.ground_truth_config.class_column_name,
            crs=self.ground_truth_config.crs,
            nproc=self.ground_truth_config.nproc,
            generate_outlines=False,
        )
        mask_generator.apply_process()

        # generate outlines
        outlines_generator = MaskOutlinesGenerator(
            rasters=self.rasters,
            output_path=self.outlines,
            output_file_prefix="outline",
            ground_truth_labels=ground_truth,
            valid_class_ids=self.ground_truth_config.valid_class_ids,
            class_column_name=self.ground_truth_config.class_column_name,
            crs=self.ground_truth_config.crs,
            nproc=self.ground_truth_config.nproc,
            generate_outlines=True,
        )
        outlines_generator.apply_process()

        # generate distance transforms
        dist_trafo_generator = DistanceTransformGenerator(
            rasters=self.rasters,
            output_path=self.distance_transforms,
            output_file_prefix="dist_trafo",
            ground_truth_labels=ground_truth,
            valid_class_ids=self.ground_truth_config.valid_class_ids,
            class_column_name=self.ground_truth_config.class_column_name,
            crs=self.ground_truth_config.crs,
            nproc=self.ground_truth_config.nproc,
        )
        dist_trafo_generator.apply_process()

    def setup(self, stage=None):  # throws error if arg is removed
        if stage == "fit":
            targets = [self.masks, self.outlines, self.distance_transforms]

            if type(targets[0]) in (list, tuple, np.ndarray):
                self.targets = [np.sort(file_list) for file_list in targets]
            else:
                self.targets = [
                    np.sort(glob.glob(os.path.abspath(file_list) + "/*.tif"))
                    for file_list in targets
                ]

            if self.shuffle:
                if self.val_indices is not None or self.train_indices is not None:
                    raise ValueError(
                        "Cannot use shuffled dataset split together with prescribed train/val indices"
                    )
                for x in (self.rasters, *self.targets):
                    np.random.shuffle(x)  # in-place

            # split into training and validation set
            data = (self.rasters, *self.targets)

            # if training and validation indices are given, use them
            if self.train_indices is None:
                training_data = [r[: int(len(r) * self.training_split)] for r in data]
            else:
                training_data = [r[self.train_indices] for r in data]

            if self.val_indices is None:
                validation_data = [r[int(len(r) * self.training_split) :] for r in data]
            else:
                validation_data = [r[self.val_indices] for r in data]

            log.info("Tiles in training data")
            for t in training_data[0]:
                log.info(t)
            log.info("Tiles in validation data")
            for t in validation_data[0]:
                log.info(t)

            # load the data into a custom dataset format
            self.train_ds = ds.TreeCrownDelineationDataset(
                training_data[0],
                training_data[1:],
                augmentation=self.augment_train,
                ndvi_config=self.ndvi_config,
                dilate_outlines=self.dilate_outlines,
                divide_by=self.divide_by,
            )

            if self.training_split < 1 or self.val_indices is not None:
                self.val_ds = ds.TreeCrownDelineationDataset(
                    validation_data[0],
                    validation_data[1:],
                    augmentation=self.augment_eval,
                    ndvi_config=self.ndvi_config,
                    dilate_outlines=self.dilate_outlines,
                    divide_by=self.divide_by,
                )

        elif stage == "test":
            if self.test_indices is not None:
                self.rasters = self.rasters[self.test_indices]
            self.test_ds = ds.TreeCrownDelineationInferenceDataset(
                self.rasters,
                augmentation=self.augment_eval,
                ndvi_config=self.ndvi_config,
                dilate_outlines=self.dilate_outlines,
                in_memory=False,
                divide_by=self.divide_by,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
