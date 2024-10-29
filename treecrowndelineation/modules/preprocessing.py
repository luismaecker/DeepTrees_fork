import os
import re

from abc import ABC
from typing import Union

import fiona

class GroundTruthGenerator(ABC):
    '''GroundTruthGenerator 

    Base class to generate ground truth (masks, outlines, distance transforms)

    Based on scripts/rasterize.py

    Loads a raster and a vector file, then rasterizes the vector file within the
    extent of the raster with the same resolution. Uses gdal_rasterize
    under the hood, but provides some more features like specifying which classes
    to rasterize into which layer of the output. If you want to infer the output
    file names, the input file name suffixes have to be delimited by an '_'.",
    '''
    def __init__(self,
                 rasters: Union[str, list],
                 output_path: str,
                 output_file_prefix: str,
                 ground_truth_labels: str,
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
            ground_truth_labels (str): Path to ground truth labels.
            nproc (int, optional): Number of parallel processes to use. Defaults to 1.
            valid_class_ids (Union[str, list]): Valid class IDs in ground_truth_labels. Defaults to 'all' (use all classes).
            class_column_name (str): Column name of class ID in ground_truth_labels.
            crs (str): Coordinate reference system. Defaults to EPSG:25832.
        '''        
        super().__init__()

        self.rasters = rasters
        self.output_path = output_path
        self.output_file_prefix = output_file_prefix
        self.ground_truth_labels = ground_truth_labels
        self.nproc = nproc

    def process(self):
        '''process
        Individual process for one tile. Needs to be defined in subclass. 
        '''
        pass

    def apply_process(self):
        '''
        Process the ground truth.
        ''' 
        with Pool(self.nprocs) as p:
            p.map(self.process, self.input_files)


class MaskOutlinesGenerator(GroundTruthGenerator):
    '''MaskOutlinesGenerator _summary_

    Args:
        GroundTruthGenerator (_type_): _description_
    '''
    def __init__(self,
                 generate_outlines: bool = False,
                 ):
        '''__init__ _summary_

        Args:
            generate_outlines (bool, optional): _description_. Defaults to False.
        '''        
        super().__init__()
        self.generate_outlines = generate_outlines

    def process(self):
        with fiona.open(self.ground_truth_labels) as src:
            # assure labels and images are in the same CRS
            assert src.crs == self.crs, \
                print(f'{self.ground_truth_labels} was expected to be {self.crs} but is {src.crs}')

            input_file = os.path.abspath(f)
            input_path, input_fname = os.path.split(input_file)

            # this is the pattern for the tiles and associated labels
            pattern = r'\d+_\d+'
            match = re.search(pattern, input_fname)
            suffix = match.group()
            #suffix = input_fname.split('.')[0].split('_')[-1]

            output_file = os.path.join(
                os.path.abspath(self.output_path),
                f'{self.output_file_prefix}_{suffix}.tif'
            )

            log.info(output_file)

            img = rioxarray.open_rasterio(f)
            bbox = extent_to_poly(img)
            features = src.filter(bbox=bbox.bounds)

            if self.generate_outlines:
                res = rasterize(img, to_outline(filter_geometry(features, args)), dim_ordering="CHW")
            else:
                res = rasterize(img, filter_geometry(features, args), dim_ordering="CHW")
            res.rio.to_raster(output_file, compress="DEFLATE")