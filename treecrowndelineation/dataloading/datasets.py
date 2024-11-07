from typing import Dict, Any
import time

import xarray as xr
import rioxarray
import numpy as np
import torch
from torchvision.transforms import v2
from numpy.typing import NDArray


# TODO check if standard map-style dataset wouldn't be sufficient here
# TODO move to random resized crop for uniform sampling
# TODO one epoch = one mini-patch out of each larger patch
from torch.utils.data import Dataset

from treecrowndelineation.modules.indices import ndvi

import logging
log = logging.getLogger(__name__)
class TreeCrownDelineationDataset(Dataset):
    """In memory remote sensing dataset for image segmentation."""
    def __init__(self, 
                 raster_files: list[str], 
                 target_files: list[str], 
                 augmentation: Dict[str, Any],
                 overwrite_nan_with_zeros: bool = True,
                 in_memory: bool = True,
                 dim_ordering="CHW",
                 dtype="float32",
                 use_weights: bool = False,
                 divide_by=1):

        """Creates a dataset containing images and targets (masks, outlines, and distance_transforms).

        Args:
            raster_files: List of file paths to source rasters. File names must be of the form '.../the_name_i.tif' where i is some index
            mask_files: A tuple containing lists of file paths to different sorts of 'masks',
                e.g. mask, outline, distance transform.
                The mask and raster file names must have the same index ending.
            TODO update docstring
            augmentation: dictionary defining augmentation 
            in_memory: If True, load full dataset into memory, else iterate. Default is True (for small labeled datasets).
            overwrite_nan_with_zeros: If True, fill NaN with 0. Default is True.
            dim_ordering: One of HWC or CHW; how rasters and masks are stored in memory. The albumentations library
                needs HWC, so this is the default. CHW support could be bugged. FIXME check if we need CHW support at all and if this is even working
            dtype: Data type for storing rasters and masks
            use_weights: If True, calculate weights according to tile size and use in loss function. Default is False.
        """
        # initial sanity checks
        assert len(raster_files) > 0, "List of given rasters is empty."
        # TODO replace this by a regex based check
        # TODO extract ID from mask
        # TODO check if that matches ID in all three targets
        for i, m in enumerate(target_files):
            if len(m) == 0:
                raise RuntimeError("Mask list {} is empty.".format(i))
            if len(m) != len(raster_files):
                raise RuntimeError("The length of the given lists must be equal.")
            for j, r in enumerate(raster_files):
                # FIXME that does not comply with our way of naming the tiles!
                raster_file_index = r.split('.')[-2].split('_')[-1]
                mask_file_index = m[j].split('.')[-2].split('_')[-1]
                if raster_file_index != mask_file_index:
                    raise RuntimeError("The raster and mask lists must be sorted equally.")

        self.raster_files = raster_files
        self.target_files = target_files
        # self.functions = functions
        self.divide_by = divide_by # TODO move this to torchvision transform
        self.augmentation = augmentation
        self.in_memory = in_memory
        self.overwrite_nan = overwrite_nan_with_zeros
        self.dtype = dtype

        self.rasters = []
        self.targets = []
        self.num_bands = 0
        self.dim_ordering = dim_ordering
        if dim_ordering == "CHW":
            self.chax = 0  # channel axis of imported arrays
        elif dim_ordering == "HWC":
            self.chax = 2
        else:
            raise ValueError("Dim ordering {} not supported. Choose one of 'CHW' or 'HWC'.".format(dim_ordering))
        self.lateral_ax = np.array((1,2)) if self.chax==0 else np.array((0,1))

        # load all rasters and targets into memory
        if self.in_memory:
            t0 = time.time()
            log.info('Loading all data into memory')
            self.load_data()
            log.info(f'Finished loading data into memory in {time.time()-t0:.1f} seconds.')

        # add augmentation functions
        transforms = [v2.ToDtype(dtype=torch.float32)]
        for key, val in self.augmentation.items():
            log.info(f'Adding augmentation {key} with parameter {val}')
            match key:
                case 'RandomResizedCrop':
                    transforms.append(v2.RandomResizedCrop(**val))
                case 'RandomCrop':
                    transforms.append(v2.RandomCrop(**val))
                case 'Resize':
                    transforms.append(v2.Resize(**val))
                case 'RandomHorizontalFlip':
                    transforms.append(v2.RandomHorizontalFlip(**val))
                case 'RandomVerticalFlip':
                    transforms.append(v2.RandomVerticalFlip(**val))
                case 'ColorJitter':
                    raise NotImplementedError('Augmentation not implemented:', key)
                case _:
                    raise ValueError(f'Augmentation not defined: {key}')
        self.augment = v2.Compose(transforms)
        
        if use_weights:
            self.weights = self.get_raster_weights()
            raise NotImplementedError('Weights are not used in loss function')
        else:
            self.weights = None  # can be used for proportional sampling of unevenly sized tiles

    # these two methods are needed for pytorch dataloaders to work
    def __len__(self):
        '''Returns length of the dataset: number of raster files'''
        return len(self.raster_files)

    def __getitem__(self, idx):
        '''__getitem__ 

        Return augmented raster and accompanying target

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        '''
        if self.in_memory: # retrieve preloaded tiles
            raster = self.rasters[idx].data
            target = self.targets[idx].data
        else: # load from disk
            raster = self.load_raster(self.raster_files[idx])
            target = self.load_target(self.target_files[idx])

        print(raster.shape)
        print(target.shape)

        # apply transforms (this ensures they are augmented in the same way)
        # FIXME check with ColorJitter ... we do not want this on targets
        raster, target = self.augment(raster, target)
        return raster, target

    def load_raster(self, file: str, used_bands: list = None):
        """Loads a raster from disk.

        Args:
            file (str): file to load
            used_bands (list): bands to use, indexing starts from 0, default 'None' loads all bands
        """
        raster = rioxarray.open_rasterio(file).load() # xarray.open_rasterio is deprecated

        if self.dim_ordering == 'CHW':
            raster = raster.transpose('band', 'y', 'x')
        elif self.dim_ordering == 'HWC':
            raster = raster.transpose('y', 'x', 'band')

        if used_bands is not None:
            raster = raster.isel(bands=used_bands)

        return raster

    def load_target(self, file: str):
        """Loads a mask from disk."""
        mask = rioxarray.open_rasterio(file).load()

        if self.dim_ordering == 'CHW':
            mask = mask.transpose('band', 'y', 'x')
        elif self.dim_ordering == 'HWC':
            mask = mask.transpose('y', 'x', 'band')

        if self.overwrite_nan:
            mask = mask.fillna(0.)

        return mask 

    def load_data(self):
        self.rasters = []
        for raster_file in self.raster_files:
            self.rasters.append(self.load_raster(raster_file))

        for files in zip(*self.target_files):
            targets = [self.load_target(f) for f in files]
            # "override" ensures that small differences in geotransorm are neglected
            target = xr.concat(targets, dim="band", join="override")
            self.targets.append(target)

    def apply_to_rasters(self, f):
        """Applies function f to all rasters."""
        for i, r in enumerate(self.rasters):
            self.rasters[i].data[:] = f(r.data).astype(self.dtype)

    def apply_to_masks(self, f):
        """Applies function f to all rasters."""
        for i, m in enumerate(self.targets):
            self.targets[i].data[:] = f(m.data).astype(self.dtype)

    def concatenate_ndvi(self, red=3, nir=4, rescale=False):
        for i, r in enumerate(self.rasters):
            # res = ndvi_xarray(r, red, nir).expand_dims(dim="band", axis=self.chax)
            if rescale:
                res = (ndvi(r, red, nir, axis=self.chax).expand_dims(dim="band", axis=self.chax) + 1) / 2
            else:
                res = ndvi(r, red, nir, axis=self.chax).expand_dims(dim="band", axis=self.chax)
            res = res.assign_coords(band=[self.num_bands + 1])  # this line took 3 hours
            self.rasters[i] = xr.concat((r, res), dim="band")
        self.num_bands += 1

    def get_raster_weights(self) -> NDArray[np.float32]:
        '''get_raster_weights 

        Calculate normalized weights according to the size of each raster tile.

        TODO maybe this should be directly torch tensor

        Returns:
            NDArray[np.float32]: array with weights per raster tile
        ''' 
        weights = [np.prod(np.array(r.shape)[self.lateral_ax]) for r in self.rasters]
        weights /= np.sum(weights)
        return weights

    def normalize(self, mean: float = 0.0, stddev: float = 1.0):
        '''normalize 

        Normalize each raster by provided mean and stddev.

        TODO add support for channel-wise mean and stddev for compliance with pretrained ViT

        Args:
            mean (float, optional): Mean to be applied. Defaults to 0.0.
            stddev (float, optional): Stddev to be applied. Defaults to 1.0.
        '''        
        f = lambda x: (x-mean)/(stddev+1E-5)
        self.apply_to_rasters(f)