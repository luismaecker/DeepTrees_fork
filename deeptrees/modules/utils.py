import os
import time
import itertools
import osgeo.gdal as gdal
import osgeo.gdalnumeric as gdn
import numpy as np
import torch
import subprocess
import fiona
import operator
from sys import stdout
from uuid import uuid4
from skimage import filters
from skimage.morphology import dilation, square, disk
from shapely.geometry import Polygon, mapping, shape
from osgeo import osr
from fiona import crs
import numpy as np
import xarray
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

import logging
log = logging.getLogger(__name__)



def overlay_heatmap(image, entropy_map, output_path, filename):
    """
    Overlay an entropy heatmap on top of the image and save the result.
    
    Parameters:
    - image: Original image (as a NumPy array or PIL image).
    - entropy_map: 2D array representing the entropy values.
    - output_path: Path to save the overlaid image.
    - filename: Name of the file to save the overlaid image.
    """
    # Ensure the image and entropy_map have compatible sizes
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = np.stack([image]*3, axis=-1)  # Make grayscale image RGB-compatible
        image_pil = Image.fromarray(image)
    else:
        image_pil = image
    
    # Create heatmap
    plt.imshow(entropy_map, cmap='jet', alpha=1)
    plt.axis('off')
    # plt.colorbar()  # Add a color bar to interpret the heatmap scale

    # Save heatmap as temporary file
    
    heatmap_path = os.path.join(output_path, 'image_heatmap.png')
    
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Load heatmap image
    heatmap = Image.open(heatmap_path).convert("RGBA")

    image_pil = np.array(image_pil)

    # Transpose the image to (height, width, channels)
    image_pil = np.transpose(image_pil, (1, 2, 0))

    # Convert to PIL image and resize to match heatmap dimensions
    image_pil = Image.fromarray((image_pil).astype(np.uint8)).resize(heatmap.size)

    # Overlay heatmap on original image
    overlaid_image = Image.blend(image_pil, heatmap, alpha=0.4)

    saved_path = os.path.join(output_path, filename)

    # Save final image with overlaid heatmap
    overlaid_image.save(saved_path)


def load_model_weights(model, path):
    """
    Loads the models weights and sets the batch norm momentum to 0.9.

    Args:
        model: The model to load weights into.
        path (str): Path to the weights file.

    Returns:
        The model with loaded weights.
    """
    model.load_state_dict(torch.load(path))
    set_batchnorm_momentum(model, 0.9)
    return model


def gpu(x: torch.Tensor, device="cuda", dtype=torch.float32):
    """
    Moves a tensor to the GPU if available.

    Args:
        x (torch.Tensor): The tensor to move.
        device (str): The device to move the tensor to.
        dtype (torch.dtype): The data type of the tensor.

    Returns:
        torch.Tensor: The tensor on the specified device.
    """
    if torch.cuda.is_available():
        return x.to(device=device, dtype=dtype)
    else:
        return x


def set_batchnorm_momentum(model, momentum):
    """
    Set the momentum for all BatchNorm layers in the given model.

    Args:
        model: The model (either scripted or non-scripted) where batchnorm layers are modified.
        momentum: The momentum value to set for the BatchNorm layers.
    """
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.momentum = momentum                


def get_map_extent(gdal_raster):
    """
    Returns a dict of {xmin, xmax, ymin, ymax, xres, yres} of a given GDAL raster file.
    Returns None if no geo reference was found.

    Args:
        gdal_raster: File opened via gdal.Open().

    Returns:
        dict: A dictionary containing the map extent.
    """
    xmin, xres, xskew, ymax, yskew, yres = gdal_raster.GetGeoTransform()
    xmax = xmin + (gdal_raster.RasterXSize * xres)
    ymin = ymax + (gdal_raster.RasterYSize * yres)
    # ret = ( (ymin, ymax), (xmin, xmax) )
    ret = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "xres": xres, "yres": yres}
    if 0. in (ymin, ymax, xmin, xmax): return None  # This is true if no real geodata is referenced.
    return ret


def gdal_trafo_to_xarray_trafo(gdal_trafo):
    """
    Converts a GDAL transform to an xarray transform.

    Args:
        gdal_trafo: The GDAL transform.

    Returns:
        tuple: The xarray transform.
    """
    xmin, xres, xskew, ymax, yskew, yres = gdal_trafo
    return (xres, xskew, xmin, yskew, yres, ymax)


def xarray_trafo_to_gdal_trafo(xarray_trafo):
    """
    Converts an xarray transform to a GDAL transform.

    Args:
        xarray_trafo: The xarray transform.

    Returns:
        tuple: The GDAL transform.
    """
    xres, xskew, xmin, yskew, yres, ymax = xarray_trafo
    return (xmin, xres, xskew, ymax, yskew, yres)


def get_xarray_extent(arr):
    """
    Returns the extent of an xarray.

    Args:
        arr: The xarray.

    Returns:
        tuple: The extent of the xarray.
    """
    xr = arr.coords["x"].data
    yr = arr.coords["y"].data
    xres, yres = (arr.transform[0], arr.transform[4])
    return min(xr), max(xr), min(yr), max(yr), xres, yres


def get_xarray_trafo(arr):
    """
    Returns the transform of an xarray.

    Args:
        arr: The xarray.

    Returns:
        tuple: The transform of the xarray.
    """
    xr = arr.coords["x"].data
    yr = arr.coords["y"].data
    xres, yres = (arr.transform[0], arr.transform[4])
    xskew, yskew = (arr.transform[1], arr.transform[3])
    return xres, xskew, min(xr), yskew, yres, max(yr)

def get_rioxarray_trafo(arr: xarray.DataArray):
    """
    Get the transform from a raster tile.

    Args:
        arr (xarray.DataArray): Raster tile.

    Returns:
        tuple: The transform of the raster tile.
    """
    xr = arr.coords['x'].data
    yr = arr.coords['y'].data
    xres = np.round(xr[1]-xr[0], 4)
    yres = np.round(yr[1]-yr[0], 4)
    xskew, yskew = (0, 0)
    return xres, xskew, min(xr), yskew, yres, max(yr)

def extent_to_poly(xarr):
    """
    Returns the bounding box of an xarray as a shapely polygon.

    Args:
        xarr: The xarray.

    Returns:
        Polygon: The bounding box polygon.
    """
    xmin, xmax, ymin, ymax, xres, yres = get_xarray_extent(xarr)
    return Polygon([(xmin, ymax), (xmin, ymin), (xmax, ymin), (xmax, ymax)])


def load_filtered_polygons(file: str,
                           rasters: list,
                           minimum_area: float = 0,
                           maximum_area: float = 10 ** 6,
                           filter_dict: dict = {},
                           operators: list = [operator.eq]
                           ) -> list:
    """
    Loads those polygons from a given shapefile which fit into the extents of the given rasters.

    Polygons will be cropped to fit the given raster extent.

    Args:
        file (str): Shapefile path.
        rasters (list): List of xarrays.
        minimum_area (float): Minimum polygon area in map units (typically m²), measured after cropping to extent.
        maximum_area (float): Maximum polygon area in map units (typically m²), measured after cropping to extent.
        filter_dict (dict): Dictionary of key value pairs to filter polygons by.
        operators (list): A list of built-in python comparison operators from the 'operator' package.

    Returns:
        list: A list of lists containing polygons in the same order as the rasters.
    """

    def filter_polygons_by_property(p):
        for i, (k, v) in enumerate(filter_dict.items()):
            val = p["properties"][k]
            if operators[i](val, v):
                return True

    fiona.supported_drivers["SQLite"] = "rw"
    polygons = []
    with fiona.open(file) as src:
        for i, r in enumerate(rasters):
            xmin, xmax, ymin, ymax, xres, yres = get_xarray_extent(r)
            bbox = (xmin, ymin, xmax, ymax)
            crop = Polygon(((xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)))

            tmp = []
            for p in src.filter(bbox=bbox):
                if len(filter_dict) > 0:
                    if not filter_polygons_by_property(p):
                        continue

                polygon = shape(p["geometry"])
                if not polygon.is_valid:
                    print("Skipping invalid polygon: {}".format(polygon))
                    continue

                intersection = crop.intersection(polygon)
                if minimum_area < intersection.area < maximum_area:
                    tmp.append(intersection)
            polygons.append(tmp)
    return polygons


def save_polygons(polygons: list, dest_fname: str, crs, driver: str = "SQLite", mode: str = "w"):
    """
    Save a list of polygons into a shapefile with given coordinate reference system.

    Args:
        polygons (list): List of shapely polygons to save.
        dest_fname (str): Path to file.
        crs: Coordinate reference system, e.g. from fiona.crs.from_epsg().
        driver (str): One of fiona's supported drivers e.g. 'ESRI Shapefile' or 'SQLite'.
        mode (str): Either 'w' for write or 'a' for append. Not all drivers support both.
    """
    fiona.supported_drivers["SQLite"] = "rw"  # ensure we can actually write to a useful format
    schema = {"geometry"  : "Polygon",
              "properties": {"id": "int"}}

    records = [{"geometry": mapping(p), "properties": {"id": i}} for i, p in enumerate(polygons)]

    if os.path.exists(dest_fname):
        uuid = str(uuid4())[:4]
        head, suff = os.path.splitext(dest_fname)
        head = head + "_"
        dest_fname_new = uuid.join([head, suff])
        print("File {} already exists, saving as {}.".format(dest_fname, dest_fname_new))
        dest_fname = dest_fname_new

    with fiona.open(dest_fname, mode, crs=crs, driver=driver, schema=schema) as f:
        f.writerecords(records)


def read_img(input_file, dim_ordering="HWC", dtype='float32', band_mapping=None, return_extent=False):
    """
    Reads an image from disk and returns it as numpy array.

    Args:
        input_file (str): Path to the input file.
        dim_ordering (str): One of HWC or CHW, C=Channels, H=Height, W=Width.
        dtype (str): Desired data type for loading, e.g. np.uint8, np.float32.
        band_mapping (dict): Dictionary of which image band to load into which array band.
        return_extent (bool): Whether or not to return the raster extent in the form (ymin, ymax, xmin, xmax).

    Returns:
        np.ndarray: Numpy array containing the image and optionally the extent.
    """
    if not os.path.isfile(input_file):
        raise RuntimeError("Input file does not exist. Given path: {}".format(input_file))

    ds = gdal.Open(input_file)
    extent = get_map_extent(ds)

    if band_mapping is None:
        num_bands = ds.RasterCount
        band_mapping = {i+1: i for i in range(num_bands)}
    elif isinstance(band_mapping, dict):
        num_bands = len(band_mapping)
    else:
        raise TypeError("band_mapping must be a dict, not {}.".format(type(band_mapping)))

    arr = np.empty((num_bands, ds.RasterYSize, ds.RasterXSize), dtype=dtype)

    for source_layer, target_layer in band_mapping.items():
        arr[target_layer] = gdn.BandReadAsArray(ds.GetRasterBand(source_layer))

    if dim_ordering == "HWC":
        arr = np.transpose(arr, (1, 2, 0))  # Reorders dimensions, so that channels are last
    elif dim_ordering == "CHW":
        pass
    else:
        raise ValueError("Dim ordering {} not supported. Choose one of 'HWC' or 'CHW'.".format(dim_ordering))

    if return_extent:
        return arr, extent
    else:
        return arr


def array_to_tif(array, dst_filename, num_bands='multi', save_background=True, src_raster: str = "", transform=None,
                 crs=None):
    """
    Takes a numpy array and writes a tif. Uses deflate compression.

    Args:
        array (np.ndarray): Numpy array.
        dst_filename (str): Destination file name/path.
        num_bands (str): 'single' or 'multi'. If 'single' is chosen, everything is saved into one layer.
        save_background (bool): Whether or not to save the last layer, which is often the background class.
        src_raster (str): Raster file used to determine the corner coords.
        transform: A geotransform in the gdal format.
        crs: A coordinate reference system as proj4 string.
    """
    if src_raster != "":
        src_raster = gdal.Open(src_raster)
        x_pixels = src_raster.RasterXSize
        y_pixels = src_raster.RasterYSize
    elif transform is not None and crs is not None:
        y_pixels, x_pixels = array.shape[:2]
    else:
        raise RuntimeError("Please provide either a source raster file or geotransform and coordinate reference "
                           "system.")

    bands = min( array.shape ) if array.ndim==3 else 1
    if not save_background and array.ndim==3: bands -= 1

    driver = gdal.GetDriverByName('GTiff')

    datatype = str(array.dtype)
    datatype_mapping = {'byte': gdal.GDT_Byte, 'uint8': gdal.GDT_Byte, 'uint16': gdal.GDT_UInt16,
                        'uint32': gdal.GDT_UInt32, 'int8': gdal.GDT_Byte, 'int16': gdal.GDT_Int16,
                        'int32': gdal.GDT_Int32, 'float16': gdal.GDT_Float32, 'float32': gdal.GDT_Float32}
    options = ["COMPRESS=DEFLATE"]
    if datatype == "float16":
        options.append("NBITS=16")

    out = driver.Create(
        dst_filename,
        x_pixels,
        y_pixels,
        1 if num_bands == 'single' else bands,
        datatype_mapping[datatype],
        options=options)

    if src_raster != "":
        out.SetGeoTransform(src_raster.GetGeoTransform())
        out.SetProjection(src_raster.GetProjection())
    else:
        out.SetGeoTransform(transform)
        srs = osr.SpatialReference()
        srs.ImportFromProj4(crs)
        out.SetProjection(srs.ExportToWkt())

    if array.ndim == 2:
        out.GetRasterBand(1).WriteArray(array)
        out.GetRasterBand(1).SetNoDataValue(0)
    else:
        if num_bands == 'single':
            singleband = np.zeros(array.shape[:2], dtype=array.dtype)
            for i in range(bands):
                singleband += (i+1)*array[:,:,i]
            out.GetRasterBand(1).WriteArray( singleband )
            out.GetRasterBand(1).SetNoDataValue(0)

        elif num_bands == 'multi':
            for i in range(bands):
                out.GetRasterBand(i+1).WriteArray( array[:,:,i] )
                out.GetRasterBand(i+1).SetNoDataValue(0)

    out.FlushCache()  # Write to disk.


def compute_pyramid_patch_weight_loss(width: int, height: int) -> np.ndarray:
    """
    Compute a weight matrix that assigns bigger weight on pixels in center and
    less weight to pixels on image boundary.
    This weight matrix is then used for merging individual tile predictions and helps dealing
    with prediction artifacts on tile boundaries.

    Args:
        width (int): Tile width.
        height (int): Tile height.

    Returns:
        np.ndarray: The weight mask.
    """
    xc = width * 0.5
    yc = height * 0.5
    xl = 0
    xr = width
    yb = 0
    yt = height

    Dcx = np.square(np.arange(width) - xc + 0.5)
    Dcy = np.square(np.arange(height) - yc + 0.5)
    Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)

    De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
    De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
    De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
    De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)

    De_x = np.sqrt(np.minimum(De_l, De_r))
    De_y = np.sqrt(np.minimum(De_b, De_t))
    De = np.minimum(De_x[np.newaxis].transpose(), De_y)

    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return W


def predict_on_array(model,
                     arr,
                     in_shape,
                     out_bands,
                     stride=None,
                     drop_border=0,
                     batchsize=64,
                     dtype="float32",
                     device="cuda",
                     augmentation=False,
                     no_data=None,
                     verbose=False,
                     report_time=False):
    """
    Applies a pytorch segmentation model to an array in a strided manner.

    Call model.eval() before use!

    Args:
        model: Pytorch model - make sure to call model.eval() before using this function!
        arr (np.ndarray): HWC array for which the segmentation should be created.
        in_shape (tuple): Input shape.
        out_bands (int): Number of output bands.
        stride (int): Stride with which the model should be applied. Default: output size.
        drop_border (int): Number of pixels to drop from the border.
        batchsize (int): Number of images to process in parallel.
        dtype (str): Desired output type (default: float32).
        device (str): Device to run on (default: cuda).
        augmentation (bool): Whether to average over rotations and mirrorings of the image or not.
        no_data: A no-data value. It's used to compute the area containing data via the first input image channel.
        verbose (bool): Whether or not to display progress.
        report_time (bool): If true, returns (result, execution time).

    Returns:
        np.ndarray: An array containing the segmentation.
    """
    t0 = None

    if augmentation:
        operations = (lambda x: x,
                      lambda x: np.rot90(x, 1),
                      # lambda x: np.rot90(x, 2),
                      # lambda x: np.rot90(x, 3),
                      # lambda x: np.flip(x,0),
                      lambda x: np.flip(x, 1))

        inverse = (lambda x: x,
                       lambda x: np.rot90(x, -1),
                       # lambda x: np.rot90(x, -2),
                       # lambda x: np.rot90(x, -3),
                       # lambda x: np.flip(x,0),
                       lambda x: np.flip(x,1))
    else:
        operations = (lambda x: x,)
        inverse = (lambda x: x,)

    assert in_shape[0] == in_shape[1], "Input shape must be equal in first two dims."
    out_shape = (in_shape[0] - 2 * drop_border, in_shape[1] - 2 * drop_border, out_bands)
    in_size = in_shape[0]
    out_size = out_shape[0]
    stride = stride or out_size
    pad = (in_size - out_size)//2
    assert pad % 2 == 0, "Model input and output shapes have to be divisible by 2."

    weight_mask = compute_pyramid_patch_weight_loss(out_size, out_size)

    original_size = arr.shape
    ymin = 0
    xmin = 0

    if no_data is not None:
        # assert arr.shape[-1]==len(no_data_vec), "Length of no_data_vec must match number of channels."
        # data_mask = np.all(arr[:,:,0].reshape( (-1,arr.shape[-1]) ) != no_data, axis=1).reshape(arr.shape[:2])
        nonzero = np.nonzero(arr[:,:,0]-no_data)
        ymin = np.min(nonzero[0])
        ymax = np.max(nonzero[0])
        xmin = np.min(nonzero[1])
        xmax = np.max(nonzero[1])
        img = arr[ymin:ymax, xmin:xmax]

    else:
        img = arr

    final_output = np.zeros(img.shape[:2]+(out_shape[-1],), dtype=dtype)

    op_cnt = 0
    for op, inv in zip(operations, inverse):
        img = op(img)
        img_shape = img.shape
        x_tiles = int(np.ceil(img.shape[1]/stride))
        y_tiles = int(np.ceil(img.shape[0]/stride))

        y_range = range(0, (y_tiles+1)*stride-out_size, stride)
        x_range = range(0, (x_tiles+1)*stride-out_size, stride)

        y_pad_after = y_range[-1]+in_size-img.shape[0]-pad
        x_pad_after = x_range[-1]+in_size-img.shape[1]-pad

        output = np.zeros( (img.shape[0]+y_pad_after-pad, img.shape[1]+x_pad_after-pad)+(out_shape[-1],), dtype=dtype)
        division_mask = np.zeros(output.shape[:2], dtype=dtype) + 1E-7
        img = np.pad(img, ((pad, y_pad_after), (pad, x_pad_after), (0, 0)), mode='reflect')

        patches = len(y_range)*len(x_range)

        def patch_generator():
            for y in y_range:
                for x in x_range:
                    yield img[y:y+in_size, x:x+in_size]

        patch_gen = patch_generator()

        y = 0
        x = 0
        patch_idx = 0
        batchsize_ = batchsize

        t0 = time.time()

        while patch_idx<patches:
            batchsize_ = min(batchsize_, patches, patches - patch_idx)
            patch_idx += batchsize_
            if verbose: log.info("\r%.2f%%" % (100 * (patch_idx + op_cnt * patches) / (len(operations) * patches)))

            batch = np.zeros((batchsize_,) + in_shape, dtype=dtype)

            for j in range(batchsize_):
                batch[j] = next(patch_gen)

            with torch.no_grad():
                prediction = model(
                    torch.from_numpy(batch.transpose((0, 3, 1, 2))).to(device=device, dtype=torch.float32))
                # prediction = torch.sigmoid(prediction)
                prediction = prediction.detach().cpu().numpy()
                prediction = prediction.transpose((0, 2, 3, 1))
            if drop_border > 0:
                prediction = prediction[:, drop_border:-drop_border, drop_border:-drop_border, :]

            for j in range(batchsize_):
                output[y:y + out_size, x:x + out_size] += prediction[j] * weight_mask[..., None]
                division_mask[y:y + out_size, x:x + out_size] += weight_mask
                x += stride
                if x + out_size > output.shape[1]:
                    x = 0
                    y += stride

        output = output / division_mask[..., None]
        output = inv(output[:img_shape[0], :img_shape[1]])
        final_output += output
        img = arr[ymin:ymax, xmin:xmax] if no_data is not None else arr
        op_cnt += 1
        if verbose: log.info("\rAugmentation step %d/%d done.\n" % (op_cnt, len(operations)))

    if verbose: stdout.flush()

    final_output = final_output/len(operations)

    if no_data is not None:
        final_output = np.pad(final_output, ((ymin, original_size[0]-ymax),(xmin, original_size[1]-xmax),(0,0)), mode='constant', constant_values=0)

    if report_time:
        return final_output, time.time() - t0

    else:
        return final_output


def predict_on_array_cf(model,
                        arr,
                        in_shape,
                        out_bands,
                        stride=None,
                        drop_border=0,
                        batchsize=64,
                        dtype="float32",
                        device="cuda",
                        augmentation=False,
                        no_data=None,
                        verbose=False,
                        aggregate_metric=False):
    """
    Applies a pytorch segmentation model to an array in a strided manner.

    Channels first version.

    Call model.eval() before use!

    Args:
        model: Pytorch model - make sure to call model.eval() before using this function!
        arr (np.ndarray): CHW array for which the segmentation should be created.
        in_shape (tuple): Input shape.
        out_bands (int): Number of output bands.
        stride (int): Stride with which the model should be applied. Default: output size.
        drop_border (int): Number of pixels to drop from the border.
        batchsize (int): Number of images to process in parallel.
        dtype (str): Desired output type (default: float32).
        device (str): Device to run on (default: cuda).
        augmentation (bool): Whether to average over rotations and mirrorings of the image or not.
        no_data: A no-data vector. Its length must match the number of layers in the input array.
        verbose (bool): Whether or not to display progress.
        aggregate_metric (bool): If true, returns (result, metric).

    Returns:
        dict: A dict containing result, time, nodata_region and time.
    """
    t0 = time.time()
    metric = 0

    if augmentation:
        operations = (lambda x: x,
                      lambda x: np.rot90(x, 1, axes=(1, 2)),
                      # lambda x: np.rot90(x, 2),
                      # lambda x: np.rot90(x, 3),
                      # lambda x: np.flip(x,0),
                      lambda x: np.flip(x, 1))

        inverse = (lambda x: x,
                   lambda x: np.rot90(x, -1, axes=(1, 2)),
                   # lambda x: np.rot90(x, -2),
                   # lambda x: np.rot90(x, -3),
                   # lambda x: np.flip(x,0),
                   lambda x: np.flip(x, 1))
    else:
        operations = (lambda x: x,)
        inverse = (lambda x: x,)

    assert in_shape[1] == in_shape[2], "Input shape must be equal in last two dims."
    out_shape = (out_bands, in_shape[1] - 2 * drop_border, in_shape[2] - 2 * drop_border)
    in_size = in_shape[1]
    out_size = out_shape[1]
    stride = stride or out_size
    pad = (in_size - out_size) // 2
    assert pad % 2 == 0, "Model input and output shapes have to be divisible by 2."

    original_size = arr.shape
    ymin = 0
    xmin = 0
    ymax = arr.shape[0]
    xmax = arr.shape[1]

    if no_data is not None:
        # assert arr.shape[-1]==len(no_data_vec), "Length of no_data_vec must match number of channels."
        # data_mask = np.all(arr[:,:,0].reshape( (-1,arr.shape[-1]) ) != no_data, axis=1).reshape(arr.shape[:2])
        nonzero = np.nonzero(arr[0, :, :] - no_data)
        if len(nonzero[0]) == 0:
            return {"prediction": None,
                    "time": time.time() - t0,
                    "nodata_region": (0, 0, 0, 0),
                    "metric": metric}

        ymin = np.min(nonzero[0])
        ymax = np.max(nonzero[0])
        xmin = np.min(nonzero[1])
        xmax = np.max(nonzero[1])
        img = arr[:, ymin:ymax, xmin:xmax]

    else:
        img = arr

    weight_mask = compute_pyramid_patch_weight_loss(out_size, out_size)
    final_output = np.zeros((out_bands,) + img.shape[1:], dtype=dtype)

    op_cnt = 0
    for op, inv in zip(operations, inverse):
        img = op(img)
        img_shape = img.shape
        x_tiles = int(np.ceil(img.shape[2] / stride))
        y_tiles = int(np.ceil(img.shape[1] / stride))

        y_range = range(0, (y_tiles + 1) * stride - out_size, stride)
        x_range = range(0, (x_tiles + 1) * stride - out_size, stride)

        y_pad_after = y_range[-1] + in_size - img.shape[1] - pad
        x_pad_after = x_range[-1] + in_size - img.shape[2] - pad

        output = np.zeros((out_bands,) + (img.shape[1] + y_pad_after - pad, img.shape[2] + x_pad_after - pad),
                          dtype=dtype)
        division_mask = np.zeros(output.shape[1:], dtype=dtype) + 1E-7
        img = np.pad(img, ((0, 0), (pad, y_pad_after), (pad, x_pad_after)), mode='reflect')

        patches = len(y_range) * len(x_range)

        def patch_generator():
            for y in y_range:
                for x in x_range:
                    yield img[:, y:y + in_size, x:x + in_size]

        patch_gen = patch_generator()

        y = 0
        x = 0
        patch_idx = 0
        batchsize_ = batchsize

        t0 = time.time()

        while patch_idx < patches:
            batchsize_ = min(batchsize_, patches, patches - patch_idx)
            patch_idx += batchsize_
            if verbose: log.info("\r%.2f%%" % (100 * (patch_idx + op_cnt * patches) / (len(operations) * patches)))

            batch = np.zeros((batchsize_,) + in_shape, dtype=dtype)

            for j in range(batchsize_):
                batch[j] = next(patch_gen)

            with torch.no_grad():
                prediction = model(torch.from_numpy(batch).to(device=device, dtype=torch.float32))
                print('batch', batch.shape)
                if aggregate_metric:
                    metric += prediction[1].cpu().numpy()
                    prediction = prediction[0]

                prediction = prediction.detach().cpu().numpy()
            if drop_border > 0:
                prediction = prediction[:, :, drop_border:-drop_border, drop_border:-dropborder]

            for j in range(batchsize_):
                output[:, y:y + out_size, x:x + out_size] += prediction[j] * weight_mask[None, ...]
                division_mask[y:y + out_size, x:x + out_size] += weight_mask
                x += stride
                if x + out_size > output.shape[2]:
                    x = 0
                    y += stride

        output = output / division_mask[None, ...]
        output = inv(output[:, :img_shape[1], :img_shape[2]])
        final_output += output
        img = arr[:, ymin:ymax, xmin:xmax] if no_data is not None else arr
        op_cnt += 1
        if verbose: log.info("\rAugmentation step %d/%d done.\n" % (op_cnt, len(operations)))

    final_output = final_output / len(operations)

    if no_data is not None:
        final_output = np.pad(final_output,
                              ((0, 0), (ymin, original_size[1] - ymax), (xmin, original_size[2] - xmax)),
                              mode='constant',
                              constant_values=0)

    return {"prediction": final_output,
            "time": time.time() - t0,
            "nodata_region": (ymin, ymax, xmin, xmax),
            "metric": metric}


def calc_band_stats(fpath : str):
    """
    Calculates the mean and standard deviation of each band in a raster file.

    Args:
        fpath (str): Path to the raster file.

    Returns:
        dict: A dictionary containing the mean and standard deviation of each band.
    """
    means = []
    stddevs = []
    p = subprocess.Popen(["gdalinfo -approx_stats {}".format(fpath)], shell=True, stdout=subprocess.PIPE)
    ret = p.stdout.readlines()
    p.wait()
    for s in ret:
        s = s.decode("UTF-8")
        if s.startswith("  Minimum"):
            m = float(s.split(",")[2].split("=")[-1])
            std = float(s.split(",")[3].split("=")[-1])
            means.append(m)
            stddevs.append(std)
    return {"mean" : np.array(means), "stddev" : np.array(stddevs)}


def dilate_img(img, size=10, shape="square"):
    """
    Dilates an image using a specified structuring element.

    Args:
        img (np.ndarray): The input image.
        size (int): The size of the structuring element.
        shape (str): The shape of the structuring element ('square' or 'disk').

    Returns:
        np.ndarray: The dilated image.
    """
    if shape == "square":
        selem = square(size)
    elif shape == "disk":
        selem = disk(size)
    else:
        ValueError("Unknown shape {}, choose 'square' or 'disk'.".format(shape))
    return dilation(img, selem)


def write_info_file(path, **kwargs):
    """
    Writes key-value pairs to a file.

    Args:
        path (str): The path to the file.
        **kwargs: Key-value pairs to write to the file.
    """
    file = open(path,'w')
    for key, value in kwargs.items():
        file.write( "{}: {}\n".format(key, value) )
    file.close()

def get_crs(array):
    """
    Retrieves the CRS from an xarray.

    Args:
        array (xarray.DataArray): The input xarray.

    Returns:
        dict: The CRS.
    """
    crs_ = array.attrs["crs"]
    if "epsg" in crs_:
        crs_ = crs.from_epsg(crs_.split(':')[-1])
    else:
        crs_ = crs.from_string(crs_)
    return crs_

def fix_crs(shape, is_crs=4326, target_crs=25832):
    """
    Fixes the CRS of a GeoDataFrame if necessary.

    Args:
        shape (gpd.GeoDataFrame): The input GeoDataFrame.
        is_crs (int): The initial CRS.
        target_crs (int): The target CRS.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with the fixed CRS.
    """
    if shape.crs is None: # naive coords
        shape.crs = is_crs

    return shape.to_crs(epsg=target_crs)


def create_batch_of_patches(input_tensor, patch_size, patch_ixs, offset, local_batch_size):
    """
    Helper function to create a batch of small patches.

    Args:
        input_tensor (torch.Tensor): Original input tensor.
        patch_size (int): Small patch size.
        patch_ixs (list): List of tuples specifying (x_start, y_start) for the small patches in the input tensor.
        offset (int): Offset of current batch in patch_ixs.
        local_batch_size (int): Size of the batch of small patches.

    Returns:
        torch.Tensor: One batch of small patches to be used in inference.
    """
    batch_of_patches = []
    for j in range(local_batch_size):
        (x_start, y_start) = patch_ixs[offset * local_batch_size + j]
        x_end   = x_start + patch_size
        y_end   = y_start + patch_size

        batch_of_patches.append(input_tensor[:, :, x_start:x_end, y_start:y_end])
        
    batch_of_patches = torch.concat(batch_of_patches)
    return batch_of_patches


def mask_and_save_individual_trees(tiff_path, polygons, output_dir, scale_factor=1):
    """
    Masks and optionally scales raster images based on given polygons.
    
    Parameters:
        tiff_path (str): Path to the input GeoTIFF file.
        polygons (list): A list of shapely geometries (polygons).
        output_dir (str): Directory to save the masked raster files.
        scale_factor (int): Factor by which to scale the output raster dimensions.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with rasterio.open(tiff_path) as src:
        for idx, polygon in enumerate(polygons):
            try:
                # Mask the raster using the current polygon's geometry
                masked_image, transform = mask(src, [mapping(polygon)], crop=True)

                # Update metadata for the output raster
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": int(masked_image.shape[1] * scale_factor),
                    "width": int(masked_image.shape[2] * scale_factor),
                    "transform": transform
                })

                # Define output file path
                output_file = os.path.join(output_dir, f"polygon_{idx}.tif")

                # Save the masked raster
                with rasterio.open(output_file, "w", **out_meta) as dst:
                    dst.write(masked_image)
                    
                print(f"Saved masked raster to: {output_file}")
            except Exception as e:
                print(f"Error processing polygon {idx}: {e}")


def predict_on_tile(model, input_tensor, patch_size=256, local_batch_size=32, stride=128):
    """
    Predict on a single tile of arbitrary dimension.

    The tile is split into smaller patches that satisfy the criterion given by the segmentation model
    that edge length must be divisible by 32. The stride parameter controls the overlap of these small patches.
    Inference is run on all small patches in a memory-efficient way. The output is collected and weighted
    with the pyramid weight function to reduce artefacts where the patches overlap.

    The output tensor contains mask, outline, and distance transform in the same shape as the input tensor.

    Args:
        model: Trained DeepTrees model.
        input_tensor (torch.Tensor): Tensor with the values of the raster tile.
        patch_size (int, optional): Patch size used in inference. Defaults to 256.
        local_batch_size (int, optional): Length of the batch of patches. Defaults to 32.
        stride (int, optional): Apply patches in a strided manner. Defaults to 128.

    Returns:
        torch.Tensor: Model output for the input raster.
    """
    # set model to evaluation mode
    model.eval()

    # retrieve input shapes
    input_shape = input_tensor.shape
    Sx = input_tensor.shape[-2] # raster edge length
    Sy = input_tensor.shape[-1] # raster edge length
    S = min(Sx, Sy) # shortest edge length

    # outputs are stored here
    output = torch.zeros(1, 3, Sx, Sy, device=model.device)
    # accumulated weight matrix for normalization
    accumulated_weight = torch.zeros(Sx, Sy, device=model.device)

    if Sx < 32 or Sy < 32:
        raise ValueError('Image smaller than 32x32 must be padded to work with SegmentationModel')

    # small patch size that is used in inference
    if S < patch_size:
        patch_size = (int(S // 32) * 32)
    patch_size = max(32, patch_size)

    # weight applied to patch
    patch_weight = torch.from_numpy(compute_pyramid_patch_weight_loss(patch_size, patch_size))
    patch_weight = patch_weight.to(model.device)

    # start and end indices of patches in x/y
    ix0_x = 0
    ix1_x = Sx - patch_size
    ix0_y = 0
    ix1_y = Sy - patch_size

    # stride defines how many patches we create
    if stride > patch_size:
        log.warning('Stride exceeded patch size, resetting to patch size')
        stride = patch_size

    # number of patches along x/y
    np_x = int(np.ceil(Sx / stride))
    np_y = int(np.ceil(Sy / stride))

    # indices of patches along x/y
    patch_ixs_x = np.linspace(ix0_x, ix1_x, np_x).astype(int)
    patch_ixs_y = np.linspace(ix0_y, ix1_y, np_y).astype(int)

    # combine in one list
    patch_ixs = list(itertools.product(patch_ixs_x, patch_ixs_y))

    # process patches as batches
    n_batches = int(np.ceil(len(patch_ixs) / local_batch_size))

    for i in range(n_batches):
        # handle last batch being shorter than the others (potentially)
        if i == n_batches - 1 and len(patch_ixs) % local_batch_size > 0:
            local_batch_size = len(patch_ixs) % local_batch_size
        # create a batch of small patches
        batch_of_patches = create_batch_of_patches(input_tensor, patch_size, patch_ixs, i, local_batch_size)

        # run inference on this batch of patches
        with torch.no_grad():
            batch_outputs = model(batch_of_patches)

        # update predictions in output array
        for j in range(local_batch_size):
            (x_start, y_start) = patch_ixs[i * local_batch_size + j]
            x_end = x_start + patch_size
            y_end = y_start + patch_size

            # add weighted patch output
            output[:, :, x_start:x_end, y_start:y_end] += batch_outputs[j] * patch_weight
            # keep track of all weights
            accumulated_weight[x_start:x_end, y_start:y_end] += patch_weight

    # divide output by weights
    output = output / accumulated_weight

    if len(input_shape) == 4:
        return output
    return output.squeeze(0)