Developer Guides
================

// Add your content here...

Installation
============



Configuration
=============


Predictions and Analysis
========================


Pre-trained models for tree segmentation
========================================



Predict on tiles
================


Tree masks
==========

Distance transform  maps
========================

You can specify to save the masks, outlines, and distance transforms for each tile. This can be informative for development purposes.


Pixel-wise entropy maps
========================

The model can output the pixel-wise entropy in inference and report the mean entropy per tile. This is used in active learning: Pixels with high entropy imply that the model was uncertain about their classification. Mostly, these pixels are at the outer boundaries of a tree crown. Use this information for active learning (see below).



Analyzing tree metrics
======================


Training
========

You can train your own model based on your own data, or finetune a pre-trained model. For this, you need to have raster tiles and the accompanying labels representing the ground truth delineated tree crowns as polygons.

This is the expected directory structure.
To train the model, you need to have the labeled tiles in the `tiles` and `labels` directories. The unlabeled tiles go into `pool_tiles`. Your polygon labels need to be in ESRI shapefile format.

```
|-- tiles
|   |-- tile_0_0.tif
|   |-- tile_0_1.tif
|   |-- ...
|-- labels
|   |-- label_tile_0_0.shp
|   |-- label_tile_0_1.shp
|   |-- ...
|-- pool_tiles
|   |-- tile_4_7.tif
|   |-- tile_4_8.tif
|   |-- ...
```

The ground truth masks, distance transforms, and outlines are created on the fly in the training script. Their directory structure is as follows:

```
|-- masks
|-- outlines
|-- dist_trafo
```

We use the following classes for training:

0 = tree
1 = cluster of trees 
2 = unsure 

By default, all classes are used for training. You can change this in the config file.


Fine-tuning pre-trained models
==============================

Adapt your own config file based on the defaults in `train_halle.yaml` as needed. For inspiration for a derived config file for finetuning, check `finetune_halle.yaml`.

Run the script like this:

```bash
python scripts/train.py # this is the default config that trains from scratch
python scripts/train.py --config-name=finetune_halle # finetune with pretrained model
python scripts/train.py --config-name=yourconfig # with your own config
```

The pretrained models need to be specified in the config file, under the attributes 'pretrained.path' for the directory containing the pretrained models, and 'pretrained.model' for the model checkpoitn filename.

You can download the pretrained model by Freudenberg et al by following the instructions above in inference.

Training your own models
========================

If you do not specify a pretrained model (pretrained.model = null), the training script will train a model from scratch. Be aware that a sizeable amount of data is needed to train deep learning models.


Prediction and Active Learning
==============================

What does the ``predict`` function do?
--------------------------------------
The ``predict`` function runs the core inference pipeline to process input imagery and generate tree crown predictions. It can also produce multiple output artifacts (e.g., uncertainty maps, tree crown masks, outlines, distance transforms, and polygons) based on the configuration settings.

How do I call the ``predict`` function?
----------------------------------------
To run predictions on a single image or a list of images, use the following command:

.. code-block:: python

   import deeptrees
   from deeptrees import predict

   predict(image_path=["path/to/image.tif"], config_path="path/to/config.yaml")

- ``image_path``: List of image file paths.
- ``config_path``: Path to the configuration file defining various settings for the prediction process.

You can specify your own model in the configuration file or use the default pre-trained model (Freudenberg et al.).

What can I extract from the ``predict`` function?
-------------------------------------------------
The primary outputs generated are:

- Uncertainty Maps (Entropy Maps): Highlight areas where the model has high uncertainty.

   - Configuration settings to enable:
     - ``active_learning: True``
     - ``save_entropy_maps: True``
     - ``entropy_maps_output_dir: "entropy_maps"``

   - Saved location: In the ``entropy_maps/`` folder as GeoTIFF files.

- Tree Crown Predictions: Includes:
   - Mask: Binary mask of predicted tree crowns.
   - Outlines: Contours of tree crowns.
   - Distance Transform: A map showing the distance to the nearest tree crown.

   - Configuration settings to enable:
     - ``save_predictions: True``
     - ``predictions_output_dir: "predictions"``

   - Saved location: In the ``predictions/`` directory.

- Individual Trees as Rasters: Extract individual trees from the prediction masks as separate raster files.

   - Configuration settings to enable:
     - ``save_masked_rasters: True``
     - ``masked_rasters_output_dir: "masked_rasters"``
     - ``scale_factor: 4``

   - Saved location: In the ``masked_rasters/`` folder.

- Polygons: Tree crown polygons saved as shapefiles (.shp) for GIS applications.

   - Configuration settings to enable:
     - ``save_polygons: True``
     - ``saved_polygons_output_dir: "saved_polygons"``

   - Saved location: In the ``saved_polygons/`` folder.

What is Active Learning, and why is it important?
-------------------------------------------------
Active learning is a process where the model identifies uncertain areas in its predictions. These areas are marked as high-entropy regions where the model is uncertain.

Entropy maps visualize where the model is uncertain by:
- Identifying regions that are ambiguous or challenging for the model.
- Allowing for focused data collection in uncertain areas, improving model performance by ensuring that low-confidence regions are labeled and incorporated into the dataset.

Since it is often not feasible to label an entire dataset, run inference on your unlabeled tiles (in ``pool_tiles``). The model will compute pixel-wise entropy and report the mean per tile. Label the tiles with the highest average entropy, then repeat fine-tuning and active learning until you achieve the desired performance.

By integrating entropy maps into your workflow, you ensure continuous improvement of the model, especially in areas where it is most likely to make errors.
