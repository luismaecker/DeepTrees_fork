from typing import Union

import os
import glob
import numpy as np
import lightning as L
import pandas as pd
import geopandas as gpd

from torch.utils.data import DataLoader
from treecrowndelineation.dataloading import datasets as ds
from treecrowndelineation.modules.utils import dilate_img, fix_crs
from treecrowndelineation.dataloading.preprocessing import MaskOutlinesGenerator, DistanceTransformGenerator

import logging
log = logging.getLogger(__name__)

class TreeCrownDelineationDataModule(L.LightningDataModule):
    def __init__(self,
                 rasters: Union[str, list],
                 masks: Union[str, list],
                 outlines: Union[str, list],
                 distance_transforms: Union[str, list],
                 ground_truth_labels: Union[str, list, None] = None,
                 training_split: float = 0.7,
                 batch_size: int = 16,
                 val_batch_size: int = 2,
                 num_workers: int = 8,
                 width: int = 256,
                 augment_train: bool = False, # TODO change type
                 augment_eval: bool = False, # TODO change type
                 concatenate_ndvi: bool = False,
                 red: int = None,
                 nir: int = None,
                 divide_by: float = 1,
                 normalize: bool = False,
                 normalization_function=None,
                 dilate_outlines: bool = False,
                 shuffle: bool = True,
                 train_indices: list[int] = None,
                 val_indices: list[int] = None,
                 rescale_ndvi: bool = True,
                 valid_class_ids: Union[str, list] = 'all',
                 class_column_name: str = 'class',
                 crs: str = 'EPSG:25832',
                 nproc: int = 1,
                 ):
        """Pytorch lightning in memory data module

        For an explanation how data loading works look at README.md in dataloading source folder.

        Args:
            rasters (str or list): Can be a list of file paths or a path to a folder containing the training raster
                files in TIF format.
            masks (str or list): List of file paths to masks, or list of masks.
            outlines (str or list): List of file paths to outlines, or list of outlines.
            distance_transforms (str or list): List of file paths to distance_transforms, or list of distance_transforms.
            ground_truth_labels (str): File or folder containing the ground truth labels.
            training_split (float): Value between 0 and 1 determining the training split. Default: 0.7
            batch_size (int): Batch size
            val_batch_size (int): Validation set batch size
            num_workers (int): Number of workers in DataLoader
            width (int): Width and height of the cropped images returned by the data loader.
            augment_train (bool): Augmentation to apply in train mode (train set).
            augment_eval (bool): Augmentation to apply in eval mode (val/test set).
            concatenate_ndvi (bool): If set to true, the NDVI (normalized difference vegetation index) will be
                appended to the rasters.You have to get the red and near IR band indices.
            red (int): Index of the red band, starting from 0.
            nir (int): Index of the near IR band, starting from 0.
            divide_by (float): Constant value to divide the rasters by. Exclusive with 'normalize' and
                'normalization_function'. Default: 1.
            normalize (bool): Normalizes the dataset to 0 mean and standard deviation 1. Exclusive with 'divide_by' and
                'normalization_function'.
            normalization_function: A function, which is applied to all images. Optional NDVI concatenation happens after
                applying the function. Exclusive with 'divide_by' and 'normalize'.
            dilate_outlines (int): The second target band (the tree outlines) can be dilated (widened) by a
                certain number of pixels.
            shuffle (bool): Whether or not to shuffle the data upon loading. This affects the partition into
                training and validation data. Default: True
            train_indices (list): (Optional) List of indices specifying which images should be assigned to the training
            set.
            val_indices: (Optional) List of indices specifying which images should be assigned to the validation
            set.
            rescale_ndvi (bool): Whether to rescale the NDVI to the interval [0,1).
        """
        super().__init__()
        if type(rasters) in (list, tuple, np.ndarray):
            self.rasters = rasters
        else:
            self.rasters = np.sort(glob.glob(os.path.abspath(rasters) + "/*.tif"))

        self.masks = masks
        self.outlines = outlines
        self.distance_transforms = distance_transforms
        self.ground_truth_labels = ground_truth_labels

        self.training_split = training_split
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.width = width
        self.augment_train = augment_train
        self.augment_eval = augment_eval
        self.concatenate_ndvi = concatenate_ndvi
        self.red = red
        self.nir = nir
        self.dilate_outlines = dilate_outlines
        self.shuffle = shuffle
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.divide_by = divide_by
        self.normalize = normalize
        self.normalization_function = normalization_function
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.rescale_ndvi = rescale_ndvi

        self.valid_class_ids = valid_class_ids
        self.class_column_name = class_column_name
        self.crs = crs
        self.nproc = nproc

        self.targets = None # will be assigned in setup_data

    def prepare_data(self) -> None:
        '''prepare_data
        
        Prepare the ground truth masks, outlines, and distance transforms from
        ground truth labels.
        '''

        if self.ground_truth_labels is None:
            log.info('No ground truth labels provided. Proceed with existing ground truth ...')
            log.info(f'Masks: {self.masks}')
            log.info(f'Outlines: {self.outlines}')
            log.info(f'Distance transforms: {self.distance_transforms}')
    
            return

        # prepare ground truth from labels
        if os.path.isfile(self.ground_truth_labels):
            ground_truth = gpd.read_file(self.ground_truth_labels)
        elif os.path.isdir(self.ground_truth_labels):
            # combine all the ground truth labels
            shapes = np.sort(glob.glob(f'{self.ground_truth_labels}/label_*.shp'))
            ground_truth = pd.concat([fix_crs(gpd.read_file(shape)).assign(tile=shape) for shape in shapes])
            log.info(f'Combining all polygons in {os.path.join(self.ground_truth_labels, 'all_labels.shp')}')
            ground_truth.drop(columns='tile').to_file(os.path.join(self.ground_truth_labels, 'all_labels.shp'))

        # generate masks
        mask_generator = MaskOutlinesGenerator(rasters=self.rasters,
                                               output_path=self.masks,
                                               output_file_prefix='mask',
                                               ground_truth_labels=ground_truth,
                                               valid_class_ids=self.valid_class_ids,
                                               class_column_name=self.class_column_name,
                                               crs=self.crs,
                                               nproc=self.nproc,
                                               generate_outlines=False)
        mask_generator.apply_process()

        # generate outlines
        outlines_generator = MaskOutlinesGenerator(rasters=self.rasters,
                                                   output_path=self.outlines,
                                                   output_file_prefix='outline',
                                                   ground_truth_labels=ground_truth,
                                                   valid_class_ids=self.valid_class_ids,
                                                   class_column_name=self.class_column_name,
                                                   crs=self.crs,
                                                   nproc=self.nproc,
                                                   generate_outlines=True)
        outlines_generator.apply_process()

        # generate distance transforms
        dist_trafo_generator = DistanceTransformGenerator(rasters=self.rasters,
                                                   output_path=self.distance_transforms,
                                                   output_file_prefix='dist_trafo',
                                                   ground_truth_labels=ground_truth,
                                                   valid_class_ids=self.valid_class_ids,
                                                   class_column_name=self.class_column_name,
                                                   crs=self.crs,
                                                   nproc=self.nproc)
        dist_trafo_generator.apply_process()

    def setup(self, stage=None):  # throws error if arg is removed
        if stage == 'fit':
            targets = [self.masks, self.outlines, self.distance_transforms]

            if type(targets[0]) in (list, tuple, np.ndarray):
                self.targets = [np.sort(file_list) for file_list in targets]
            else:
                self.targets = [np.sort(glob.glob(os.path.abspath(file_list) + "/*.tif")) for file_list in targets]

            if self.shuffle: # FIXME shuffle should not be used together with fixed train indices!
                for x in (self.rasters, *self.targets):
                    np.random.shuffle(x)  # in-place

            # split into training and validation set
            data = (self.rasters, *self.targets)

            # if traiing and validation indices are given, use them
            if self.train_indices is not None:
                training_data = [r[self.train_indices] for r in data]
            else:
                training_data = [r[:int(len(r) * self.training_split)] for r in data]

            if self.val_indices is not None:
                validation_data = [r[self.val_indices] for r in data]
            else:
                validation_data = [r[int(len(r) * self.training_split):] for r in data]

            log.info('Tiles in training data')
            for t in training_data[0]:
                log.info(t)
            log.info('Tiles in validation data')
            for t in validation_data[0]:
                log.info(t)

            # load the data into a custom dataset format
            self.train_ds = ds.TreeCrownDelineationDataset(training_data[0],
                                                    training_data[1:],
                                                    augmentation=self.augment_train,
                                                    divide_by=self.divide_by)

            # TODO do we ever need anything besides divide_by == 1
            # TODO if (!!) at all we should provide external mean/std eg for ViT
            if sum([self.divide_by != 1, self.normalization_function is not None, self.normalize]) > 1:
                raise RuntimeError("Please provide either 'divide_by', 'normalize' or 'normalization_function' as argument.")
            elif self.normalization_function is not None:
                self.train_ds.apply_to_rasters(self.normalization_function)
            elif self.normalize:
                self.train_ds.normalize()

            if self.training_split < 1 or self.val_indices is not None:
                self.val_ds = ds.TreeCrownDelineationDataset(validation_data[0],
                                                        validation_data[1:],
                                                        augmentation=self.augment_eval,
                                                        divide_by=self.divide_by)

                if self.normalization_function is not None:
                    self.val_ds.apply_to_rasters(self.normalization_function)
                elif self.normalize:
                    self.val_ds.normalize()

            # attach the NDVI to the rasters
            if self.concatenate_ndvi and self.red is not None and self.nir is not None:
                # we rescale the NDVI to [0...1] to allow gamma augmentation to work right
                self.train_ds.concatenate_ndvi(red=self.red, nir=self.nir, rescale=self.rescale_ndvi)
                if self.training_split < 1 or self.val_indices is not None:
                    self.val_ds.concatenate_ndvi(red=self.red, nir=self.nir, rescale=self.rescale_ndvi)

            # dilate the tree crown outlines to get a stronger training signal
            if self.dilate_outlines:
                for m in self.train_ds.masks:
                    m[:, :, 1] = dilate_img(m[:, :, 1], self.dilate_outlines)
                if self.training_split < 1 or self.val_indices is not None:
                    for m in self.val_ds.masks:
                        m[:, :, 1] = dilate_img(m[:, :, 1], self.dilate_outlines)
        
        elif stage == 'test':
            self.test_ds = ds.TreeCrownDelineationDataset(training_data[0],
                                                    training_data[1:],
                                                    augmentation=self.augment_eval,
                                                    divide_by=self.divide_by)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        if self.training_split == 1 and self.val_indices is None:
            return None
        else:
            return DataLoader(self.val_ds, batch_size=self.val_batch_size, num_workers=self.num_workers, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False)