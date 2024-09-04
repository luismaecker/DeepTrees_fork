# Training

This document outlines the steps for fine-tuning a pretrained TreeCrownDelineation model on Levante.

## Directory structure

The root folder is `/work/ka1176/shared_data/2024-ufz-deeptree/polygon-labelling/`. Sync this folder with the labeled tiles provided by UFZ.

```
|-- tiles
|   |-- tile_0_0.tif
|   |-- tile_0_1.tif
|   |-- ...
|-- labels
|   |-- label_tile_0_0.shp
|   |-- label_tile_0_1.shp
|   |-- ...
```

Create the new empty directories

```
|-- masks
|-- outlines
|-- dist_trafo
```

## Preparation

We will follow the instructions in the TreeCrownDelineation repository to fine tune the models. Link: https://github.com/AWF-GAUG/TreeCrownDelineation

1. Combine all labels into one shapefile `all_labels.shp`. Make sure the coordinate reference system is `EPSG:25832` to comply with the tiles.

```python
import glob
import pandas as pd
import geopandas as gpd

shapes = np.sort(glob.glob('/work/ka1176/shared_data/2024-ufz-deeptree/polygon-labelling/labels/label_tile_*.shp'))
all_polygons = pd.concat([gpd.read_file(shape).set_crs(epsg=4326).to_crs(epsg=25832) for shape in shapes])
all_polygons.to_file('/work/ka1176/shared_data/2024-ufz-deeptree/polygon-labelling/labels/all_labels.shp')
```

2. Create the raster image tiles: skip, they are provided by UFZ (TODO: Check)

3. Rasterize the delineated tree crowns. We are working in `~treecrowndelineation/scripts`.

```python
python rasterize.py -i /work/ka1176/shared_data/2024-ufz-deeptree/polygon-labelling/tiles/* -o /work/ka1176/shared_data/2024-ufz-deeptree/polygon-labelling/masks/mask_ -shp /work/ka1176/shared_data/2024-ufz-deeptree/polygon-labelling/labels/all_labels.shp
```

4. Create the outlines.

```python
python rasterize.py -i /work/ka1176/shared_data/2024-ufz-deeptree/polygon-labelling/tiles/* -o /work/ka1176/shared_data/2024-ufz-deeptree/polygon-labelling/outlines/outline_ -shp /work/ka1176/shared_data/2024-ufz-deeptree/polygon-labelling/labels/all_labels.shp --outlines
```

5. Create the distance transform.

```python
python rasterize_to_distance_transform.py -i /work/ka1176/shared_data/2024-ufz-deeptree/polygon-labelling/tiles/ -o /work/ka1176/shared_data/2024-ufz-deeptree/polygon-labelling/dist_trafo/dist_trafo_ -shp /work/ka1176/shared_data/2024-ufz-deeptree/polygon-labelling/labels/all_labels.shp
```

6. Check that everything was processed correctly. Run the notebook `notebooks/processing/quick_data_check.ipynb` for a visual inspection.