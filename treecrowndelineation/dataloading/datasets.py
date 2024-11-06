import xarray as xr
import numpy as np
from numpy.typing import NDArray

from torch.utils.data import IterableDataset

from treecrowndelineation.modules.indices import ndvi
class TreeCrownDelineationDataset(IterableDataset):
    """In memory remote sensing dataset for image segmentation."""
    def __init__(self, 
                 raster_files: list[str], 
                 mask_files: list[str], 
                 augmentation, 
                 cutout_size, # FIXME duplicate with the cutout size defined for albumentation, mover here?? 
                 overwrite_nan_with_zeros: bool = True,
                 in_memory: bool = True,
                 dim_ordering="HWC",
                 dtype="float32", 
                 divide_by=1):

        """Creates a dataset containing images and masks which resides in memory.

        Args:
            raster_files: List of file paths to source rasters. File names must be of the form '.../the_name_i.tif' where i is some index
            mask_files: A tuple containing lists of file paths to different sorts of 'masks',
                e.g. mask, outline, distance transform.
                The mask and raster file names must have the same index ending.
            TODO update docstring
            in_memory: If True, load full dataset into memory, else iterate. Default is True (for small labeled datasets).
            overwrite_nan_with_zeros: If True, fill NaN with 0. Default is True.
            dim_ordering: One of HWC or CHW; how rasters and masks are stored in memory. The albumentations library
                needs HWC, so this is the default. CHW support could be bugged. FIXME check if we need CHW support at all and if this is even working
            dtype: Data type for storing rasters and masks
        """
        # initial sanity checks
        assert len(raster_files) > 0, "List of given rasters is empty."
        for i, m in enumerate(mask_files):
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
        self.mask_files = mask_files
        # self.functions = functions
        self.divide_by = divide_by
        self.augment = augmentation  # or (lambda x, y: (x, y))
        self.cutout_size = cutout_size
        self.in_memory = in_memory
        self.overwrite_nan = overwrite_nan_with_zeros
        self.dtype = dtype

        self.rasters = []
        self.masks = []
        self.num_bands = 0
        self.num_mask_bands = 0
        self.dim_ordering = dim_ordering
        self.native_bands = 0
        self.weights = None  # can be used for proportional sampling of unevenly sized tiles
        if dim_ordering == "CHW":
            self.chax = 0  # channel axis of imported arrays
        elif dim_ordering == "HWC":
            self.chax = 2
        else:
            raise ValueError("Dim ordering {} not supported. Choose one of 'CHW' or 'HWC'.".format(dim_ordering))
        self.lateral_ax = np.array((1,2)) if self.chax==0 else np.array((0,1))

        if self.in_memory:
            self.load_data()

    # these two methods are needed for pytorch dataloaders to work
    def __len__(self):
        # sum of product of all raster sizes
        total_pixels = np.sum([np.prod(np.array(r.shape)[self.lateral_ax]) for r in self.rasters])
        # product of the shape of cutout done by the transformation
        # uses albumentation augmentation API
        cutout_pixels = np.prod(np.array(self.cutout_size)[self.lateral_ax])
        return int(total_pixels / cutout_pixels)

    def __iter__(self):
        i = 0
        while i < len(self):
            idx = np.random.choice(np.arange(len(self.rasters)), p=self.weights)
            augmented = self.augment(image=self.rasters[idx].data,
                                     mask=self.masks[idx].data)  # giving the data only should speed things up!
            image = augmented["image"].transpose((2, 0, 1))
            mask = augmented["mask"].transpose((2, 0, 1))
            i += 1
            yield image, mask

    def load_raster(self, file: str, used_bands: list = None):
        """Loads a raster from disk.

        Args:
            file (str): file to load
            used_bands (list): bands to use, indexing starts from 0, default 'None' loads all bands
        """
        arr = xr.open_rasterio(file).load().astype(self.dtype)  # eagerly load the image from disk via_load
        arr.close()  # dont know if needed, but to be sure...

        num_bands = arr.shape[0]
        if len(self.rasters) == 0:
            self.num_bands = num_bands
        if self.num_bands != num_bands:
            raise ValueError(
                "Number of raster layers ({}) does not match previous raster layer count ({}).".format(num_bands,
                                                                                                       self.num_bands))

        if used_bands is not None:
            arr = arr[used_bands]

        if self.dim_ordering == "HWC":
            arr = arr.transpose('y', 'x', 'band')

        # TODO FIXME what does he mean by that?
        arr.data /= self.divide_by  # dividing the array directly loses information on transformation etc?!?? wtf?

        self.rasters.append(arr)
        self.native_bands = np.arange(self.num_bands)
        self.weights = self.get_raster_weights()

    def load_mask(self, file: str):
        """Loads a mask from disk."""
        arr = xr.open_rasterio(file).load().astype(self.dtype)  # eagerly load the image from disk via load
        arr.close()  # dont know if needed, but to be sure...

        num_bands = arr.shape[0]
        if len(self.masks) == 0: self.num_mask_bands = num_bands

        if self.dim_ordering == "HWC":
            arr = arr.transpose('y', 'x', 'band')

        if self.overwrite_nan:
            nanmask = np.isnan(arr.data)
            arr.data[nanmask] = 0

        return arr

    def load_data(self):
        for r in self.raster_files:
            self.load_raster(r)

        for files in zip(*self.mask_files):
            masks = [self.load_mask(f) for f in files]
            # "override" ensures that small differences in geotransorm are neglected
            mask = xr.concat(masks, dim="band", join="override")
            self.masks.append(mask)

    def apply_to_rasters(self, f):
        """Applies function f to all rasters."""
        for i, r in enumerate(self.rasters):
            self.rasters[i].data[:] = f(r.data).astype(self.dtype)

    def apply_to_masks(self, f):
        """Applies function f to all rasters."""
        for i, m in enumerate(self.masks):
            self.masks[i].data[:] = f(m.data).astype(self.dtype)

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

    def get_raster_weights(self) -> NDArray[float]:
        '''get_raster_weights 

        Calculate normalized weights according to the size of each raster tile.

        Returns:
            _type_: _description_
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