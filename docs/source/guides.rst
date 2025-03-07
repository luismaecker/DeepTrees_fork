Developer Guides
================

Configuring DeepTrees
=====================

DeepTrees is a modular software that can be configured to suit your specific needs. The configuration file is a YAML file that defines various settings for the training, prediction, and evaluation processes. The configuration file is divided into sections, each corresponding to a specific aspect of the software. Arguments for the datamodule, the model, and the training / inference process are specified there.

DeepTrees uses `Hydra <https://hydra.cc/docs/intro/>`_ for configuration management. Hydra allows you to compose and override configurations flexibly. The configuration files are located in the `configs` directory. You can create your own configuration file or modify the existing ones to suit your needs.

By default, the following configuration files are used with the training / inference scripts:

.. code-block::

  Predicting with package (deeptrees.predict) > configs/predict/inference_on_individual_tiles.yaml

  Training / inference with python scripts:

  train.py > configs/train/train_halle.yaml
  test.py > configs/test/inference_halle.yaml


To override the default configuration settings, you should pass your configuration file path as an argument to the training or inference scripts. 

Hydra configs are composable, meaning you do not need to specify parameters again that have been specified in the basis configuration file. Plus, you can overwrite individual configuration parameters on the command line. 



Analyzing tree metrics
======================



Prediction
==========

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

Labeling data efficiently
=========================

Creating the ground truth segmentation polygons is a time-consuming process. DeepTree implements active learning to help you direct your labeling efforts to the most informative regions of the dataset.

What is Active Learning, and why is it important?
-------------------------------------------------
Active learning is a process where the model identifies uncertain areas in its predictions. These areas are marked as high-entropy regions where the model is uncertain.

Entropy maps visualize where the model is uncertain by:
- Identifying regions that are ambiguous or challenging for the model.
- Allowing for focused data collection in uncertain areas, improving model performance by ensuring that low-confidence regions are labeled and incorporated into the dataset.

Since it is often not feasible to label an entire dataset, run inference on your unlabeled tiles (in ``pool_tiles``). The model will compute pixel-wise entropy and report the mean per tile. Label the tiles with the highest average entropy, then repeat fine-tuning and active learning until you achieve the desired performance.

By integrating entropy maps into your workflow, you ensure continuous improvement of the model, especially in areas where it is most likely to make errors.

Dataset 
=======

Use your own dataset
--------------------

The data is handled by the `TreeCrownDelineationDataModule` and the `TreeCrownDelineationBaseDataset`. This class provides functions to load the data, preprocess it, and return it in a format that can be used by the model.

DeepTrees can process raster tiles in `TIF` format, e.g. from digital orthophotos. You can provide your own dataset by replacing the corresponding paths `data.rasters` in the configuration file.

If you want to provide imagery in a different format, you can modify the `TreeCrownDelineationBaseDataset` class to handle the data accordingly.

DeepTrees comes with a small dataset for demonstration purposes.

6. **Create ground truth for training and validation**

For training or validating with your own dataset, you will create ground truth tree crown polygons in an external software, e.g. QGIS. We work with the following classes:

- `0`: Tree
- `1`: Cluster of trees
- `2`: Unsure
- `3`: (dead tree, not yet implemented)

The deep learning model requires the ground truth tree crown polygons to be transformed into raster masks, distance transforms, and outlines. These can be created on the fly during training or inference.

Option 1, if you use the provided script `train.py` together with a configuration file derived from `configs/train.yaml`: Set `data.ground_truth_config.labels` to the path of the directory containing the shapefiles with the polygons. During setup of the datamodule, the target masks, distance transforms, and outlines will be created.

Option 2, if you want to generate the target masks, distance transforms, and outlines stand-alone: 

.. code-block::

  from deeptrees.dataloading.datamodule import TreeCrownDelineationDataModule

  tcdm = TreeCrownDelineationDataModule(**config)
  tcdm.prepare_data()

Check `configs/train.yaml` and the `TreeCrownDelineationDataModule` class for an example configuration.

Data Augmentation
-----------------

The DeepTrees dataset class provides data augmentation options, which can be enabled in the configuration file (`data.augment_train`, `data.augment_eval`). The following torchvision augmentations are available:

- Random resized crop
- Resize 
- Random crop
- Random horizontal flip
- Random vertical flip

To add more augmentations, you can modify the `TreeCrownDelineationBaseDataset` class. Augmentations need to be based on torchvision v2 transforms to work with the current augmentation pipeline.

NDVI Calculation and other indices
----------------------------------

The Normalized Difference Vegetation Index (NDVI) is a common index used to assess vegetation health and density. You can add the NDVI band to your dataset by setting the `data.ndvi_config.concatenate = True` in the configuration file. 

Note that this attaches the NDVI to your other input channels and needs to be reflected in your model's number of input channels. To add more indices, you can modify the `TreeCrownDelineationBaseDataset` class.

Training
========


You can train your own model based on your own data, or finetune a pre-trained model. For this, you need to have raster tiles and the accompanying labels representing the ground truth delineated tree crowns as polygons.

This is the expected directory structure.
To train the model, you need to have the labeled tiles in the `tiles` and `labels` directories. The unlabeled tiles go into `pool_tiles`. Your polygon labels need to be in ESRI shapefile format.

.. code-block::

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

The ground truth masks, distance transforms, and outlines are created on the fly in the training script. Their directory structure is as follows:

.. code-block::

    |-- masks
    |-- outlines
    |-- dist_trafo

We use the following classes for training:

0 = tree
1 = cluster of trees 
2 = unsure 

By default, all classes are used for training. You can change this in the config file.


Fine-tune a pretrained model
----------------------------

Starting from a pretrained model that can be downloaded in `datasets` (see above), you can finetune the model on your own data. This is currently handled by the `train.py` script. It supports starting the training with weights from a pretrained model.

The pretrained model should be passed in `data.pretrained.path` (root folder) and `data.pretrained.model` (checkpoint file). For inspiration for a configuration file, check `configs/train/finetune_halle.yaml`.

Run the training script like this:

.. code-block::

  python scripts/train.py --config-name=finetune_halle # finetune with pretrained model (demo for the Halle DOP dataset)
  python scripts/train.py --config-name=yourconfig # with your own config

Train a model from scratch
--------------------------

If you do not specify a pretrained model (`pretrained.model = null` in the configuration file), the training script will train a model from scratch. Be aware that a sizeable amount of data is needed to train deep learning models.

Control the training loop
-------------------------

DeepTrees is a modular software based to large parts on `Pytorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_ modules. Training is handled by the Lightning Trainer. To control aspects of the training loop, modify the `trainer` section in the configuration file based on the Lightning Trainer API.

Model architectures
===================

TreeCrownDelineationModel
-------------------------

We currently support the `TreeCrownDelineationModel`, following the implementation by Freudenberg et al, as a backbone to the `DeepTreesModel`. 


Add your own model
------------------

Thanks to the modular structure, it is easy to substitute your own model architecture. Add your own model to the repository and make sure it inherits from Lightning Module. Then, modify the `DeepTreesModel` in `models.py` to use your new model as a backbone, instead of `TreeCrownDelineationModel`. Add your model's keyword arguments to the configuration file. It will be instantiated while running the `train.py` script. 

Be aware that novel models will not work with the pretrained model weights.