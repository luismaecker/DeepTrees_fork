.. DeepTrees documentation master file, created by
   sphinx-quickstart on Sun Feb 23 13:05:05 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DeepTrees🌳
=======================
**Tree Crown Segmentation and Analysis in Remote Sensing Imagery with PyTorch**


Welcome to the DeepTrees documentation! 

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   overview
   configuration
   guides
   deeptrees

DeepTrees is a PyTorch-based library for end-to-end tree crown segmentation and analysis in multispectral remote sensing imagery. 

Highlights:
	-	User-friendly, flexible, and GPU-optimized for efficiency.
	-	Supports 4-channel imagery (RGBi) with built-in PyTorch data loaders.
	-	Simple API for training and evaluating U-Net-based tree segmentation and distance transform models.
	-	Generates pixel-entropy maps for active learning and fine-tuning.
	-	Computes vegetation indices, tree crown health and allometric metrics.
	-	Pre-trained models and sample labeled datasets for DOP imagery in Central Germany.
	-	Easily configurable via a YAML file.
	-	Integration of Geospatial Foundation Model (GeoFM) backbones.
	-	Tree species classification and height estimation models.


DeepTrees is a result of the `DeepTrees <https://deeptrees.de>`_ project, a collaboration between the Helmholtz Center for Environmental Research -- UFZ and the Helmholtz AI initiative.

Installation
=======

From PyPi registry:

.. code-block:: bash

    pip install deeptrees

or from source:

.. code-block:: bash

    git clone https://codebase.helmholtz.cloud/taimur.khan/DeepTrees.git
    python3 setup.py install


Cite As
=======

If you use DeepTrees in your research, please cite it as follows:

.. admonition:: APA Citation

   Khan, T., Arnold, C., & Grover, H. (2025). *DeepTrees: Tree crown segmentation and analysis in remote sensing imagery with PyTorch*. arXiv. `https://doi.org/10.48550/arXiv.XXXXX.YYYYY <https://doi.org/10.48550/arXiv.XXXXX.YYYYY>`_

.. admonition:: BibTeX Citation

   .. code-block:: bibtex

      @article{khan2025deeptrees,
        author    = {Taimur Khan and Caroline Arnold and Harsh Grover},
        title     = {DeepTrees: Tree Crown Segmentation and Analysis in Remote Sensing Imagery with PyTorch},
        journal   = {arXiv},
        year      = {2025},
        archivePrefix = {arXiv},
        eprint    = {XXXXX.YYYYY},  
        primaryClass = {cs.CV}      
      }

License
=======

This package is license under the MIT License. See the `LICENSE <https://codebase.helmholtz.cloud/taimur.khan/DeepTrees/-/blob/main/LICENSE?ref_type=heads>`_ file for details.