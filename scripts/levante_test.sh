#!/bin/bash

source ~/.bashrc
conda activate deeptree

# Reference run with the old inference script
TILEROOT=/work/ka1176/shared_data/2024-ufz-deeptree/Halle-DOP20-2022/
TILE=dop20rgbi_32_702_5706_2_st_2022.tif
PRETRAINED_0="/work/ka1176/shared_data/2024-ufz-deeptree/pretrained_models/tcd-20cm-RGBI-v1/Unet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=0_jitted.pt"
OUTPUTROOT=./tmp/
OUTPUT_0=${OUTPUTROOT}/${TILE/.tif}_0.sqlite
python inference.py -i ${TILEROOT}/${TILE} -o ${OUTPUT_0} -m ${PRETRAINED_0} --ndvi --red 0 --nir 3 --divide-by 255

# inference with new script but old model
python scripts/inference_halle.py 'model_path="/work/ka1176/shared_data/2024-ufz-deeptree/pretrained_models/tcd-20cm-RGBI-v1/Unet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=0_jitted.pt"'

# inference with new script and new model
python scripts/inference_halle.py model_path=/work/ka1176/shared_data/2024-ufz-deeptree/finetuned_models/demo-unet-halle-jitted.pt
