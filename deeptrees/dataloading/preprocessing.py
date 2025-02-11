'''
Classes to be used in preprocessing labels -> ground truth rasters

Based on scripts/rasterize.py and scripts/rasterize_to_distance_transform.py
'''
import os
import re

from abc import ABC
from typing import Union

import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray
from shapely import Polygon
from scipy.ndimage import distance_transform_edt
from multiprocessing import Pool

from ..modules import rasterize_utils as rutils

import logging
log = logging.getLogger(__name__)
class GroundTruthGenerator(ABC):
    '''GroundTruthGenerator 

    Base class to generate ground truth (masks, outlines, distance transforms)

    Loads a raster and a vector file, then rasterizes the vector file within the
    extent of the raster with the same resolution. Uses gdal_rasterize
    under the hood, but provides some more features like specifying which classes
    to rasterize into which layer of the output. If you want to infer the output
    file names, the input file name suffixes have to be delimited by an '_'.",

    Based on scripts/rasterize.py in Freudenberg 2022.
    '''
    def __init__(self,
                 rasters: Union[str, list],
                 output_path: str,
                 output_file_prefix: str,
                 ground_truth_labels: Union[str, gpd.GeoDataFrame],
                 valid_class_ids: Union[str, list] = 'all',
                 class_column_name: str = 'class', # TODO
                 crs: str = 'EPSG:25832',
                 nproc: int = 1,
                 ):
        '''__init__ 

        Args:
            rasters (Union[str, list]): (List of) file path(s) to the raster files
            output_path (str): Output directory
            output_file_prefix (str): Output file prefix. Suffix is infered from raster files.
            ground_truth_labels (Union[str, gpd.GeoDataFrame]): Path to ground truth labels or frame with labels
            nproc (int, optional): Number of parallel processes to use. Defaults to 1.
            valid_class_ids (Union[str, list]): Valid class IDs in ground_truth_labels. 
                Defaults to 'all' (use all classes).
            class_column_name (str): Column name of class ID in ground_truth_labels.
            crs (str): Coordinate reference system. Defaults to EPSG:25832.
        '''        
        super().__init__()

        self.rasters = rasters
        self.output_path = output_path
        self.output_file_prefix = output_file_prefix
        self.valid_class_ids = valid_class_ids
        self.class_column_name = class_column_name
        self.crs = crs
        self.nproc = nproc

        if isinstance(ground_truth_labels, str):
            self.ground_truth_labels = gpd.read_file(ground_truth_labels)
        else:
            self.ground_truth_labels = ground_truth_labels

    def output_filename(self, input_file: str) -> str:
        '''setup_process 

        - Construct output file name from input file name

        Args:
            input_file (str): input file

        Returns:
            str: output file
        '''

        input_file = os.path.abspath(input_file)
        _, input_fname = os.path.split(input_file)

        # this is the pattern for the tiles and associated labels
        pattern = r'\d+_\d+' # TODO this may change in the future
        match = re.search(pattern, input_fname)
        suffix = match.group()
        #suffix = input_fname.split('.')[0].split('_')[-1]

        output_file = os.path.join(
            os.path.abspath(self.output_path),
            f'{self.output_file_prefix}_{suffix}.tif'
        )

        return output_file

    def constrain_geometry_to_tile(self, input_file: str) -> list[Polygon]:
        '''constrain_geometry_to_raster 

        Filter the ground truth labels that fall within the bounding box of the
        given raster image.

        Args:
            input_file (str): path to input raster file

        Returns:
            list[Polygon]: list of polygons within tile bounding box
        '''
        # assure labels and images are in the same CRS
        if self.crs != str(self.ground_truth_labels.crs):
            raise ValueError(f'CRS was expected to be {self.crs} but is {self.ground_truth_labels.crs}')

        bbox = rutils.get_bbox_polygon(input_file)
        # constrain to current tile
        features = self.ground_truth_labels[self.ground_truth_labels.intersects(bbox)]
        # filter for valid classes
        features = rutils.filter_geometry(features, self.valid_class_ids, self.class_column_name)

        return features

    def process(self, input_file: str):
        '''
        Process function that works on one tile. Needs to be defined in subclass.

        Args:
          input_file (str): Path to input input_file file.
        '''
        pass

    def apply_process(self):
        '''
        Apply the processing function in parallel.
        '''
        with Pool(self.nproc) as p:
            p.map(self.process, self.rasters)

class MaskOutlinesGenerator(GroundTruthGenerator):
    '''MaskOutlinesGenerator

    Generate masks and outlines from tiles and ground truth labels.
    '''
    def __init__(self,
                 rasters: Union[str, list],
                 output_path: str,
                 output_file_prefix: str,
                 ground_truth_labels: Union[str, gpd.GeoDataFrame],
                 valid_class_ids: Union[str, list] = 'all',
                 class_column_name: str = 'class', # TODO
                 crs: str = 'EPSG:25832',
                 nproc: int = 1,
                 generate_outlines: bool = False,
                 ):
        '''
        Initialize the MaskOutlinesGenerator instance.

        Args:
            rasters (Union[str, list]): (List of) file path(s) to the raster files
            output_path (str): Output directory
            output_file_prefix (str): Output file prefix. Suffix is infered from raster files.
            ground_truth_labels (Union[str, gpd.GeoDataFrame]): Path to ground truth labels or frame with labels
            nproc (int, optional): Number of parallel processes to use. Defaults to 1.
            valid_class_ids (Union[str, list]): Valid class IDs in ground_truth_labels. 
              Defaults to 'all' (use all classes).
            class_column_name (str): Column name of class ID in ground_truth_labels.
            crs (str): Coordinate reference system. Defaults to EPSG:25832.
            generate_outlines (bool): If True, generate outlines. If False, generate masks. 
              Defaults to False.
        '''
        super().__init__(rasters, output_path, output_file_prefix, ground_truth_labels, valid_class_ids, 
                         class_column_name, crs, nproc)
        self.generate_outlines = generate_outlines

    def process(self, input_file: str):
        '''process Create raster mask/outline from input tile

        Args:
            input_file (str): input raster file
        '''
        img = rioxarray.open_rasterio(input_file)
        output_file = self.output_filename(input_file)
        features = self.constrain_geometry_to_tile(input_file)

        if self.generate_outlines:
            res = rutils.rasterize(img, rutils.to_outline(features), dim_ordering="CHW")
        else:
            res = rutils.rasterize(img, features, dim_ordering="CHW")
        res.rio.to_raster(output_file, compress="DEFLATE")

class DistanceTransformGenerator(GroundTruthGenerator):
    '''DistanceTransformGenerator

    Generate distance transforms from tiles and ground truth labels.
    '''
    def __init__(self,
                 rasters: Union[str, list],
                 output_path: str,
                 output_file_prefix: str,
                 ground_truth_labels: Union[str, gpd.GeoDataFrame],
                 valid_class_ids: Union[str, list] = 'all',
                 class_column_name: str = 'class', # TODO
                 crs: str = 'EPSG:25832',
                 nproc: int = 1,
                 area_max: int = None,
                 area_min: int = 3
                 ):
        '''
        Initialize the MaskOutlinesGenerator instance.

        Args:
            rasters (Union[str, list]): (List of) file path(s) to the raster files
            output_path (str): Output directory
            output_file_prefix (str): Output file prefix. Suffix is infered from raster files.
            ground_truth_labels (Union[str, gpd.GeoDataFrame]): Path to ground truth labels or frame with labels
            nproc (int, optional): Number of parallel processes to use. Defaults to 1.
            valid_class_ids (Union[str, list]): Valid class IDs in ground_truth_labels. 
              Defaults to 'all' (use all classes).
            class_column_name (str): Column name of class ID in ground_truth_labels.
            crs (str): Coordinate reference system. Defaults to EPSG:25832.
            area_max (int): Maximum area of polygons to consider. Defaults to None.
            area_min (int): Minimum area of polygons to consider. Defaults to 3.
        '''
        super().__init__(rasters, output_path, output_file_prefix, ground_truth_labels, valid_class_ids, 
                         class_column_name, crs, nproc)
        self.area_max = area_max
        self.area_min = area_min
        
    def process(self, input_file):
        '''process Create raster distance_transform from input tile

        The distance transform is normalized per polygon (better for instance 
        segmentation).

        Args:
            input_file (str): input raster file
        '''

        img = rioxarray.open_rasterio(input_file)
        output_file = self.output_filename(input_file)
        features = self.constrain_geometry_to_tile(input_file)
        mask = xr.zeros_like(img[[0]]).astype("float32")  # dirty hack to get three layers

        for i, polygon in enumerate(features):
            if self.area_max is not None and polygon.area > self.area_max:
                log.info(f'skipping polygon {i} because it is too large with area {polygon.area}')
                continue

            if polygon.area < self.area_min:
                log.info(f'skipping polygon {i} because it is too small with area {polygon.area}')
                continue

            # restrict to rectangle bounding box of current polygon
            xmin_p, ymin_p, xmax_p, ymax_p = polygon.bounds
            polygon_area = mask.loc[:, ymax_p:ymin_p, xmin_p:xmax_p].astype("float32")
            if 0 in polygon_area.shape: # polygon below resolution
                continue

            # mask for current polygon
            rasterized = rutils.rasterize(polygon_area, [polygon], dim_ordering="CHW")[0]

            # calculate distance transform
            padded = np.pad(rasterized,1)
            distance_transformed = distance_transform_edt(padded)[1:-1,1:-1].astype("float32")
            distance_transformed /= max(np.max(distance_transformed), 1)
            polygon_area[0] = distance_transformed
            # distance transform added to output mask
            mask.loc[:, ymax_p:ymin_p, xmin_p:xmax_p] += polygon_area

        # clip excess distances
        mask[0] = np.clip(mask[0], 0, 1)
        mask.rio.to_raster(output_file, compress="DEFLATE")
