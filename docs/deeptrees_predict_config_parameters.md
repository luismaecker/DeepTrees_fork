# Predict with DeepTrees: Configuration Parameters

The default configuration parameters are given in (round brackets).

## Data Configuration

- `data` 
  - `ndvi_config` **NDVI Configuration**
    - `concatenate`: Concatenate NDVI as a 5th channel (True)
    - `rescale`: Rescale NDVI values to [0,1] range (False)
    - `red`: Index of red channel (0)
    - `nir`: Index of near-infrared channel (3)
  - `divide_by`: Input normalization factor (255)
  - `dilate_outlines`: Dilate prediction outlines by number of pixels (0)

## Model Configuration
- `model`
  - `num_backbones`: Number of backbone networks (1)
  - `in_channels`: Number of input image channels (5)
  - `architecture`: Neural network architecture (Unet)
  - `backbone`: Backbone network (ResNet18)
  - `apply_sigmoid`: Apply sigmoid activation to outputs (False)

## Pretrained Model
- `download_pretrained_model`: If True, download the pretrained model weights (True)
- `pretrained_model_path`: Directory for pretrained models (*./pretrained/*)
- `pretrained_model_name`: Filename of pretrained model (*lUnet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=3_jitted.pt*)

## Polygon Extraction Parameters
- `min_dist`: Minimum pixel distance between local maxima (10)
- `mask_exp`: Feature map extraction exponent (2)
- `outline_multiplier`: Outline enhancement factor (5)
- `outline_exp`: Outline feature exponent (1)
- `dist_exp`: Distance transform exponent (0.5)
- `area_min`: Minimum polygon area threshold (3)
- `sigma`: Gaussian filter standard deviation (2)
- `label_threshold`: Local maxima height threshold (0.5)
- `binary_threshold`: Feature map background threshold (0.1)
- `simplify`: Polygon vertex simplification distance (0.3)

## Output Options
- `active_learning`: Enable entropy heatmap calculation
- `save_entropy_maps`: Save entropy heatmap outputs
- `save_predictions`: Save prediction masks and transforms
- `save_masked_rasters`: Save individual tree raster masks
- `masked_rasters_output_dir`: Output directory for masked rasters
- `scale_factor`: Output raster scaling factor (4)

## Geopandas Information
- `crs`: Coordinate reference system (EPSG:25832)
- `polygon_file`: Output polygon shapefile name