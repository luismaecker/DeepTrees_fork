
<a href="https://www.helmholtz.ai" target="_blank">
<img src="static/hai.png" alt="HelmholtzAI" align="right" height="60px" style="margin-top: 0; margin-right: 30px" />
</a>
<a href="https://www.ufz.de" target="_blank">
<img src="static/ufz.png" alt="UFZLogo" align="right" height="60px" style="margin-top: 0; margin-right: 10px" />
</a>
<br/>
<br/>

<div align="left" style="text-align:left">
<h1>DeepTrees 🌳</h1>
</div>
<div align="center" style="text-align:center">
  <h3>Tree Crown Segmentation and Analysis in Remote Sensing Imagery with PyTorch</h3> 
  <div align="center">
  <a href="https://badge.fury.io/py/deeptrees">
    <img src="https://badge.fury.io/py/deeptrees.svg" alt="PyPI version">
  </a>
  <a href="http://dx.doi.org/10.13140/RG.2.2.32837.36329">
  <img src="http://img.shields.io/badge/DOI-10.13140/RG.2.2.32837.36329-19C3AD.svg" alt="DOI">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://codebase.helmholtz.cloud/taimur.khan/DeepTrees/-/pipelines">
    <img src="https://codebase.helmholtz.cloud/taimur.khan/DeepTrees/badges/main/pipeline.svg" alt="CI Build">
  </a>
  <br/>
</div> 
  <img src="./static/deeptrees.png" alt="DeepTrees" width=800"/>
  <br/>
  <br/>
</div>

DeepTrees is a end-to-end library for tree crown semantic and instance segmentation, as well as analysis in remote sensing imagery. It provides a modular and flexible framework based on PyTorch for training, active-learning and deploying deep learning models for tree crown semantic and instance segmentation. The library is designed to be easy to use and extendable, with a focus on reproducibility and scalability. It includes a variety of pre-trained models, datasets, and tree allometrical metrics to help you understand tree crown dynamics. 

Read more about this work and find tutorials on: [https://deeptrees.de](https://deeptrees.de). The DeepTrees project is funded by the Helmholtz Centre for Environmental Research -- UFZ, in collaboration with Helmholtz AI.

## Installation

To install the package, clone the repository and install the dependencies.

```bash
git clone https://codebase.helmholtz.cloud/taimur.khan/DeepTrees.git
cd DeepTrees

pip install -r requirements.txt
```

or from Gitlab registry:

```bash
pip install deeptrees --index-url https://codebase.helmholtz.cloud/api/v4/projects/13888/packages/pypi/simple
```


or from PyPI.

```bash
pip install deeptrees
```

> Note: DeepTrees uses python libaries that depend on GDAL. Make sure to have GDAL>=3.9.2 installed on your system, e.g. via conda: `conda install -c conda-forge gdal==3.9.2`. 

## API Documentation

You can view the documentation page on: [https://deeptrees.readthedocs.io/en/latest/](https://deeptrees.readthedocs.io/en/latest/)

This library is documented using Sphinx. To build the documentation, run the following command.

```bash
sphinx-apidoc -o docs/source deeptrees 
cd docs
make html
```

This will create the documentation in the `docs/build` directory. Open the `index.html` file in your browser to view the documentation.

## Configuration

This software requires Hydra for configuration management. The configuration **yaml** files are stored in the `config` directory. 

This software uses Hydra for configuration management. The configuration files are stored in the `config` directory. 

The confirguration schema can be found in the `config/schema.yaml` file.

A list of Hydra configurations can be found in: [/docs/prediction_config.md](/docs/prediction_config.md)

## Pretrained Models

DeepTrees provides a set of pretrained models for tree crown segmentation. Currently the following models are available:

| Author | Description   | Model Weights |
|--------|---------------|---------------|
| [Freudenberg et al., 2022](https://doi.org/10.1007/s00521-022-07640-4) | Tree Crown Delineation model based on U-Net with ResNet18 backbone. Trained on 89 images sampled randomly within Germany. Set of 5 model weights from 5-fold cross validation. | k=[0](https://syncandshare.desy.de/index.php/s/NcFgPM4gX2dtSQq/download/lUnet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=0_jitted.pt), [1](https://syncandshare.desy.de/index.php/s/NcFgPM4gX2dtSQq/download/lUnet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=1_jitted.pt), [2](https://syncandshare.desy.de/index.php/s/NcFgPM4gX2dtSQq/download/lUnet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=2_jitted.pt), [3 (default)](https://syncandshare.desy.de/index.php/s/NcFgPM4gX2dtSQq/download/lUnet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=3_jitted.pt), [4](https://syncandshare.desy.de/index.php/s/NcFgPM4gX2dtSQq/download/lUnet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=4_jitted.pt) |
| | |

> Note: We are in the process of adding more pretrained models.

> Note: Like all AI systems, these pretrained models can make mistakes.  Validate predictions, especially in critical applications. Be aware that performance may degrade significantly on data that differs from the training set (e.g., different seasons, regions, or image qualities)

Download the pretrained models from the links:

```python
from deeptrees.pretrained import freudenberg2022

freudenberg2022(
  filename="name_your_file", # name of the file to save the model
  k=0, # number of k-fold cross validation
  return_dict=True # returns the weight pytorch model weights dictionary
)
```

## Datasets

DeepTrees also provides a lablled DOP dataset with DOP20cm rasters and corresponding polygon labels as `.shp` files. The dataset is available for download:

```python
from deeptrees.datasets.halleDOP20 import load_tiles, load_labels

load_tiles(zip_filename="path/to/tiles.zip") #give the path to where you want to save the tiles
load_labels(zip_filename="path/to/labels.zip") #give the path to where you want to save the labels
```
> Note: We are in the process of adding more datasets and updating the current datasets.

## Predict on a list of images

Predict tree crown polygons for a list of images. The configuration file in `config_path` controls the pretrained model, output paths, and postprocessing options.

```bash
from deeptrees import predict

predict(image_path=["list of image_paths"],  config_path = "config_path")
```

## Training

DeepTrees calculates pixel-wise entropy maps for each input image. The entropy maps can be used to select the most informative tiles for training. The entropy maps are stored in the `entropy` directory.


To train the model, you need to have the labeled tiles in the `tiles` and `labels` directories. The unlabeled tiles go into `pool_tiles`. Your polygon labels need to be in ESRI shapefile format.

Adapt your own config file based on the defaults in `train_halle.yaml` as needed. For inspiration for a derived config file for finetuning, check `finetune_halle.yaml`.

Run the script like this:

```bash
python scripts/train.py # this is the default config that trains from scratch
python scripts/train.py --config-name=finetune_halle # finetune with pretrained model
python scripts/train.py --config-name=yourconfig # with your own config
```

To re-generate the ground truth for training, make sure to pass the label directory in `data.ground_truth_labels`. To turn it off, pass `data.ground_truth_labels=null`.

You can overwrite individual parameters on the command line, e.g.

```bash
python scripts/train.py trainer.fast_dev_run=True
```

To resume training from a checkpoint, take care to pass the hydra arguments in quotes to avoid the shell intercepting the string (pretrained model contains `=`):

```bash
python scripts/train.py 'model.pretrained_model="Unet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=0_jitted.pt"'
```

#### Expected Directory structure

Before you embark onSync the folder `tiles` and `labels` with the labeled tiles. The unlabeled tiles go into `pool_tiles`.

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

Create the new empty directories

```
|-- masks
|-- outlines
|-- dist_trafo
```

### Training Classes

We use the following classes for training:

0 = tree
1 = cluster of trees 
2 = unsure 
3 = dead trees (haven’t added yet)

However, you can adjust classes as needed in your own training workflow.



### Training Logs

By default, [MLFlow](https://mlflow.org/) logs are created during training.

### Inference

Run the inference script with the corresponding config file. Adjust as needed.

```bash
python scripts/test.py --config-name=inference_halle
```


## Semantic Versioning
This repository has auto semantic versionining enabled. To create new releases, we need to merge into the default `main` branch. 

Semantic Versionining, or SemVer, is a versioning standard for software ([SemVer website](https://semver.org/)). Given a version number MAJOR.MINOR.PATCH, increment the:

- MAJOR version when you make incompatible API changes
- MINOR version when you add functionality in a backward compatible manner
- PATCH version when you make backward compatible bug fixes
- Additional labels for pre-release and build metad

See the SemVer rules and all possible commit prefixes in the [.releaserc.json](.releaserc.json) file. 

| Prefix | Explanation                                                                                                                                                                                                                                     | Example                                                                                              |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| feat   | A new feature was implemented as part of the commit, <br>so the [Minor](https://mobiuscode.dev/posts/Automatic-Semantic-Versioning-for-GitLab-Projects/#minor) part of the version will be increased once <br>this is merged to the main branch | feat: model training updated                                            |
| fix    | A bug was fixed, so the [Patch](https://mobiuscode.dev/posts/Automatic-Semantic-Versioning-for-GitLab-Projects/#patch) part of the version will be <br>increased once this is merged to the main branch                                         | fix: fix a bug that causes the user to not <br>be properly informed when a job<br>finishes |

The implementation is based on. https://mobiuscode.dev/posts/Automatic-Semantic-Versioning-for-GitLab-Projects/


# License

This repository is licensed under the MIT License. For more information, see the [LICENSE.md](LICENSE.md) file.

# Cite as

```bib
@article{khan2025deeptrees,
        author    = {Taimur Khan and Caroline Arnold and Harsh Grover},
        title     = {DeepTrees: Tree Crown Segmentation and Analysis in Remote Sensing Imagery with PyTorch},
        year      = {2025},
        journal   = {ResearchGate},
        archivePrefix = {ResearchGate},
        eprint    = {10.13140/RG.2.2.32837.36329},
        doi    = {http://dx.doi.org/10.13140/RG.2.2.32837.36329},  
        primaryClass = {cs.CV}      
      }
```