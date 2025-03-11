Architectural Overview
======================

DeepTrees realeases started with version 1.0.0 because it is a fork of the TreeCrownDelineation (TCD) library that has been highly restructured and extended to a full PyTorch-based library for a more general use of the software for model training, feature extraction, evaluation, predictions, and analysis. DeepTrees especially focuses on the analysis part as the authors believe that current deep learning implementations for tree crown delineation do not provide comprehensive metrics for downstream applications in ecology, forestry, urban planning and biodiversity research.
The TCD library was developed by Maximillian Freudenberg et al. at the University of Göttingen and can be found at: `https://github.com/AWF-GAUG/TreeCrownDelineation <https://github.com/AWF-GAUG/TreeCrownDelineation>`_. 

The TCD model incorporates two U-Net models for tree crown segmentation and distance transform prediction. The U-Net models are trained on a dataset of 4-channel imagery (RGBi) with corresponding ground truth labels. The distance transform model is used to predict the distance of each pixel to the nearest tree crown boundary. The distance transform model is trained on the same dataset as the segmentation model, but with the distance transform labels. A complete overview of the TCD architecture can be found in the `TCD paper <https://doi.org/10.1007/s00521-022-07640-4>`_.

.. image:: _static/tcd.png
    :alt: Architectural Diagram
    :width: 600px
    :align: center




DeepTrees uses the distance transform model to compute the pixel-entropy map, which is used for active learning and fine-tuning of the segmentation model. The pixel-entropy map is computed by taking the entropy of the distance transform predictions for each pixel. The pixel-entropy map is used to identify pixels that are close to the tree crown boundary, but are not accurately predicted by the segmentation model. These pixels are then used to retrain the segmentation model. DeepTrees also computes certain tree traits and offer user to mask the tree crowns in the image for further analysis.
A full overview of the deeptrees architecture can be found in the figure below:

.. image:: https://codebase.helmholtz.cloud/taimur.khan/DeepTrees/-/raw/7db84d5f0e99a07b144bca2f67d591d5ea0c8501/static/deeptrees.png
    :alt: Pixel Entropy Map
    :width: 600px
    :align: center
