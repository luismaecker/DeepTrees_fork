<div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
  <a href="https://www.ufz.de" target="_blank">
    <img src="static/ufz.png" alt="UFZLogo" height="90px" style="margin-top: 0; margin-left: 10px" />
  </a>
  <a href="https://www.helmholtz.ai" target="_blank">
    <img src="static/hai.png" alt="HelmholtzAI" height="70px" style="margin-top: 0; margin-right: 30px" />
  </a>
</div>
<hr/>

<div align="center" style="text-align:center">
  <h1>DeepTrees 🌳</h1>
  <h3>Tree Crown Segmentation and Analysis in Remote Sensing Imagery with PyTorch</h3>  
  <br/>
  <img src="./static/header.png" alt="DeepTrees" width="300"/>
  <br/>
</div>

<div align="center">
  <a href="https://badge.fury.io/py/deeptrees">
    <img src="https://badge.fury.io/py/deeptrees.svg" alt="PyPI version">
  </a>
  <a href="https://doi.org/10.5281/zenodo.5555555">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.5555555.svg" alt="DOI">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://codebase.helmholtz.cloud/taimur.khan/DeepTrees/-/pipelines">
    <img src="https://codebase.helmholtz.cloud/taimur.khan/DeepTrees/badges/main/pipeline.svg" alt="CI Build">
  </a>
</div>

## Installation

To install the package, clone the repository and install the dependencies.

```bash
git clone https://codebase.helmholtz.cloud/ai-consultants-dkrz/DeepTrees.git
cd DeepTrees

## create a new conda environment
pip install -r requirements.txt
```

or from pip.

```bash
pip install deeptrees
```

## Documentation

You can view the documentation on: https://taimur.khan.pages.hzdr.de/deeptrees

This library is documented using Sphinx. To build the documentation, run the following command.

```bash
sphinx-apidoc -o docs/source deeptrees 
cd docs
make html
```

This will create the documentation in the `docs/build` directory. Open the `index.html` file in your browser to view the documentation.

## Predict on a list of images

Run the inference script with the corresponding config file on list of images.

```bash
from deeptrees import predict

predict(image_path=["list of image_paths"],  config_path = "config_path")
```

## Configuration

This software uses Hydra for configuration management. The configuration files are stored in the `config` directory. 

The configuration schema can be found in
This software uses Hydra for configuration management. The configuration files are stored in the `config` directory. 

The confirguration schema can be found in the `config/schema.yaml` file.

You can find sample configuration files for training and prediction in the following links:
- [Training Configs](https://taimur.khan.pages.hzdr.de/deeptrees/config/train/)
- [Prediction Configs](https://taimur.khan.pages.hzdr.de/deeptrees/config/predict/)

A list of prediction configurations can be found in: [/docs/prediction_config.md](/docs/prediction_config.md)


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



#### Training Logs

View the MLFlow logs that were created during training.

TODO

### Inference

Run the inference script with the corresponding config file. Adjust as needed.

```bash
python scripts/test.py --config-name=inference_halle
```


## Semantic Versioning
This reposirotry has auto semantic versionining enabled. To create new releases, we need to merge into the default `finetuning-halle` branch. 

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
@article{khan2025torchtrees,
        author    = {Taimur Khan and Caroline Arnold and Harsh Grover},
        title     = {DeepTrees: Tree Crown Segmentation and Analysis in Remote Sensing Imagery with PyTorch},
        journal   = {arXiv},
        year      = {2025},
        archivePrefix = {arXiv},
        eprint    = {XXXXX.YYYYY},  
        primaryClass = {cs.CV}      
      }
```