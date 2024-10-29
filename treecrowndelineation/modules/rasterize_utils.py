''' 
Utils for creating ground truth raster files from polygons.
'''

from typing import Dict, Union

import numpy as np
import xarray as xr
import geopandas as gpd
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
from osgeo import gdalnumeric as gdn

from shapely.geometry import shape, Polygon

def get_xarray_extent(xarr: xr.DataArray) -> Dict[str, float]:
    '''get_xarray_extent

    Get the extent of an xarray DataArray.

    Args:
        xarr (xr.DataArray): Input array. 

    Returns:
        Dict[str, float]: Dictionary with extents.
    '''        
    x = xarr.coords["x"].data
    y = xarr.coords["y"].data
    gt = [float(x) for x in xarr.spatial_ref.GeoTransform.split()]
    xres, yres = (gt[1], gt[5])

    xdict = {'xmin': min(x), 'xmax': max(x),
             'ymin': min(y), 'ymax': max(x),
             'xres': xres, 'yres': yres}
    return xdict

def extent_to_poly(xarr: xr.DataArray) -> Polygon:
    '''extent_to_poly
    Returns the bounding box of an xarray as 
    shapely polygon.

    Args:
        xarr (xr.DataArray): Input array.

    Returns:
        Polygon: Bounding box of input array.
    '''

    xdict = get_xarray_extent(xarr)
    return Polygon([(xdict['xmin'], xdict['ymax']),
                    (xdict['xmin'], xdict['ymin']),
                    (xdict['xmax'], xdict['ymin']),
                    (xdict['xmax'], xdict['ymax']),
                    ])

def xarray_trafo_to_gdal_trafo(xarray_trafo):
    xres, xskew, xmin, yskew, yres, ymax = xarray_trafo
    return (xmin, xres, xskew, ymax, yskew, yres)


def get_xarray_trafo(arr):
    """Returns
    xmin, xmax, ymin, ymax, xres, yres
    of an xarray. xres and yres can be negative.
    """
    x = arr.coords["x"].data
    y = arr.coords["y"].data
    gt = [float(x) for x in arr.spatial_ref.GeoTransform.split()]
    xres, yres = (gt[1], gt[5])
    xskew, yskew = (gt[2], gt[4])
    return xres, xskew, min(x), yskew, yres, max(y)




def rasterize(source_raster, features: list, dim_ordering: str = "HWC"):
    """ Rasterizes the features (polygons/lines) within the extent of the given xarray with the same resolution, all in-memory.

    Args:
        source_raster: Xarray
        features: List of shapely objects
        dim_ordering: One of CHW (default) or HWC (height, widht, channels)
    Returns:
        Rasterized features
    """
    ncol = source_raster.sizes["x"]
    nrow = source_raster.sizes["y"]

    # Fetch projection and extent
    if "crs" in source_raster.attrs:
        proj = source_raster.attrs["crs"]
    else:
        proj = source_raster.rio.crs.to_proj4()

    ext = xarray_trafo_to_gdal_trafo(get_xarray_trafo(source_raster))

    raster_driver = gdal.GetDriverByName("MEM")
    out_raster_ds = raster_driver.Create('', ncol, nrow, 1, gdal.GDT_Byte)
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)

    spatref = osr.SpatialReference()
    spatref.ImportFromProj4(proj)

    vector_driver = ogr.GetDriverByName("Memory")
    vector_ds = vector_driver.CreateDataSource("")
    vector_layer = vector_ds.CreateLayer("", spatref, ogr.wkbMultiLineString)
    defn = vector_layer.GetLayerDefn()

    for poly in features:
        feature = ogr.Feature(defn)
        geom = ogr.CreateGeometryFromWkb(poly.wkb)
        feature.SetGeometry(geom)
        vector_layer.CreateFeature(feature)

    vector_layer.SyncToDisk()

    gdal.RasterizeLayer(out_raster_ds,
                        [1],
                        vector_ds.GetLayer(),
                        burn_values=[1],
                        options=['ALL_TOUCHED=TRUE']
                        )

    out_raster_ds.FlushCache()
    bands = [out_raster_ds.GetRasterBand(i) for i in range(1, out_raster_ds.RasterCount + 1)]
    arr = xr.zeros_like(source_raster[[0],:,:])
    arr[:] = np.array([gdn.BandReadAsArray(band) for band in bands]).astype(np.uint8)
    arr.attrs["nodatavals"] = (0,)
    arr.attrs["scales"] = (1,)
    arr.attrs["offsets"] = (0,)

    if dim_ordering == "HWC":
        arr = arr.transpose((1, 2, 0))
    del out_raster_ds
    del vector_ds
    return arr


def filter_geometry(polygons: gpd.GeoDataFrame,
                    valid_classes: Union[str, list] = 'all',
                    class_column_name: str = 'class') -> list[Polygon]:
    '''filter_geometry

    Filter the provided polygons by keeping only valid classes.

    Args:
        polygons (gpd.GeoDataFrame): GeoDataFrame containing the polygons and class labels.
        valid_classes (Union[str, list]): List of valid class labels. Defaults to 'all' (use all classes).
        class_column_name (str): Column name of class labels in src. Defaults to 'class'.

    Returns:
        list[Polygon]: filtered list of Polygons
    '''    

    filtered_polygons = []
    for i in range(len(polygons)):
        if valid_classes == 'all' or polygons[class_column_name].iloc[i] in valid_classes:
            filtered_polygons.append(polygons['geometry'].iloc[i])
    return filtered_polygons

def to_outline(polygons: list[Polygon]):
    '''to_outline

    Args:
        polygons (list[Polygon]): list of polygons

    Returns:
        _type_: TODO type list of boundaries of the polygons
    '''   
    return (p.boundary for p in polygons)
