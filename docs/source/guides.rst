Developer Guides
===============

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
=====================


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

Active learning
===============

It is often not feasible to label your whole dataset. Active learning is a technique to help you identify the samples that are most informative for your model.

Run inference on your unlabeled tiles (in pool_tiles). The model will predict the pixel wise entropy and report its mean per tile. The tiles with the highest average entropy are the tiles that you should label next. Repeat finetuning and the active learning loop, until you are satisfied with the results.

Training your own models
========================

If you do not specify a pretrained model (pretrained.model = null), the training script will train a model from scratch. Be aware that a sizeable amount of data is needed to train deep learning models.


Prediction
==========

To run predictions on a single image or a list of images, use the predict function. 
Pass the image paths as a list along with a corresponding configuration file. 
Adapt your own config file based on the defaults in `inference_on_individual_tiles.yaml` as needed. 

You can specify your own model in the configuration file. By default, a pre-trained model (ref Freudenberg et al) will be used.

Example usage::

    from deeptrees import predict

    predict(image_path=["path/to/image.tif"], config_path="path/to/config.yaml")

