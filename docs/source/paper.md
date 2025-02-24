---
title: "TorchTrees: Tree Crown Segmentation and Analysis in Remote Sensing Imagery with PyTorch"
authors:
  - name: "Taimur Khan"
    affiliation: 1
    email: "taimur.khan@ufz.de"
  - name: "Caroline Arnold"
    affiliation: 23
    email: "caroline.arnold@hereon.de"
  - name: "Harsh Grover"
    affiliation: 23
    email:"harsh.Grover@hereon.de"
affiliations:
  - index: 1
    name: "Helmholtz Center for Environmental Research -- UFZ"
    address: "Permoserstr. 15, 04318 Leipzig, Germany"
  - index: 2
    name: "Helmholtz-Zentrum Dresden-Rossendorf (HZDR)"
  - index: 3
    name: "Helmholtz AI initiative"
date: 2025-01-01
bibliography: paper.bib
---

## Summary

TorchTrees is a Python package for tree crown segmentation and analysis in remote sensing imagery. It uses PyTorch for training and predicting on large-scale image datasets. Designed for direct integration with geospatial workflows, TorchTrees provides data loaders, transforms, and model utilities, enabling efficient experimentation in tree canopy segmentation, allometrical traits analysis, and tree vitality estimation.

## Statement of Need

Accurate tree crown segmentation is essential for ecological modeling, biomass estimation, and forest management. Traditional methods often depend on labor-intensive manual delineation or specialized scripts. With the proliferation of high-resolution imagery from satellite and UAV platforms, there is a pressing need for a scalable, open-source tool that can:

- Automate the segmentation of tree crowns across diverse landscapes.
- Seamlessly integrate with geospatial workflows for data loading, tiling, and inference.
- Provide robust methods for crown morphological analysis and distance transform calculations.
- Support reproducible research with transparent and customizable training pipelines.

While other deep learning-based tree detection and segmentation tools focus primarily on these tasks, TorchTrees distinguishes itself as an end-to-end solution that also emphasizes analysis. This comprehensive approach facilitates downstream ecological tasks. Key features include:

- U-Net-based segmentation masks and distance transform models.
- Pixel-entropy maps for uncertainty estimation.
- Active learning strategies for model improvement.
- Customizable training configurations via YAML files.
- Get started quickly with pre-trained models and datasets.
- Get maps and masks at every stage of the pipeline.
- Tree health, allometric traits, and power-law calculations.
- GPU-accelerated training and inference.

TorchTrees also leverages geospatial foundation models to enhance its segmentation capabilities. These models, pre-trained on extensive geospatial datasets, provide a robust starting point for fine-tuning on specific tree crown delineation tasks. By incorporating geospatial foundation models, TorchTrees ensures higher accuracy and generalization across various landscapes and imaging conditions.

The modular design of TorchTrees supports data scientists and environmental researchers who require an open-source tool to manage large-scale aerial, UAV, or satellite datasets.

## Main Features

- Integrated data loaders for RGBi imagery
- GPU-accelerated PyTorch pipelines
- Automatic logging of training metrics
- Tree health, allometric traits, and power-law calculations

## Datasets and models

TorchTrees includes pre-trained models on the [TreeCrownDelineation dataset](https://github.com/AWF-GAUG/TreeCrownDelineation). 

Additionally, the package also provides a labelled dataset of tree crowns in the Halle region as ESRI shape files, which can be used for training and evaluation. The treecrowns are labelled with the following classes:

0 = tree
1 = cluster of trees
2 = unsure
3 = dead trees (haven’t added yet)

## Acknowledgements

This work is part of the DeepTrees project, funded by the Helmholtz Center for Environmental Research and supported by the Helmholtz AI initiative.

## References

- [1] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems (NeurIPS), 2019. https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library
- [2] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, 2015. https://doi.org/10.1007/978-3-319-24574-4_28
- [3] Freudenberg, M., Magdon, P., & Nölke, N. (2022). Individual tree crown delineation in high-resolution remote sensing images based on U-Net. Neural Computing and Applications (NCAA), Springer, 2022. https://doi.org/10.1007/s00521-022-07640-4.
