#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --time=00:30:00
#SBATCH --mem=50G
#SBATCH -G 1
#SBATCH -o slurm/slurm-%j.out

source ~/.bashrc
conda activate tree2

TILEROOT=/work/ka1176/shared_data/2024-ufz-deeptree/Halle-DOP20-2022/
TILE=${1} # pass filename as argument, e.g. dop20rgbi_32_704_5708_2_st_2022.tif

PRETRAINED_0="/work/ka1176/shared_data/2024-ufz-deeptree/pretrained_models/tcd-20cm-RGBI-v1/Unet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=0_jitted.pt"
PRETRAINED_1="/work/ka1176/shared_data/2024-ufz-deeptree/pretrained_models/tcd-20cm-RGBI-v1/Unet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=1_jitted.pt"
PRETRAINED_2="/work/ka1176/shared_data/2024-ufz-deeptree/pretrained_models/tcd-20cm-RGBI-v1/Unet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=2_jitted.pt"
PRETRAINED_3="/work/ka1176/shared_data/2024-ufz-deeptree/pretrained_models/tcd-20cm-RGBI-v1/Unet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=3_jitted.pt"
PRETRAINED_4="/work/ka1176/shared_data/2024-ufz-deeptree/pretrained_models/tcd-20cm-RGBI-v1/Unet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=4_jitted.pt"

OUTPUTROOT=/work/ka1176/shared_data/2024-ufz-deeptree/Halle-out-TCD-pretrained-rgbi

OUTPUT_0=${OUTPUTROOT}/${TILE/.tif}_0.sqlite
OUTPUT_1=${OUTPUTROOT}/${TILE/.tif}_1.sqlite
OUTPUT_2=${OUTPUTROOT}/${TILE/.tif}_2.sqlite
OUTPUT_3=${OUTPUTROOT}/${TILE/.tif}_3.sqlite
OUTPUT_4=${OUTPUTROOT}/${TILE/.tif}_4.sqlite

# Uses RGBI channels
# Uses NDVI
# Default settings
python inference.py -i ${TILEROOT}/${TILE} -o ${OUTPUT_0} -m ${PRETRAINED_0} --ndvi --red 0 --nir 3 --divide-by 255
python inference.py -i ${TILEROOT}/${TILE} -o ${OUTPUT_1} -m ${PRETRAINED_1} --ndvi --red 0 --nir 3 --divide-by 255
python inference.py -i ${TILEROOT}/${TILE} -o ${OUTPUT_2} -m ${PRETRAINED_2} --ndvi --red 0 --nir 3 --divide-by 255
python inference.py -i ${TILEROOT}/${TILE} -o ${OUTPUT_3} -m ${PRETRAINED_3} --ndvi --red 0 --nir 3 --divide-by 255
python inference.py -i ${TILEROOT}/${TILE} -o ${OUTPUT_4} -m ${PRETRAINED_4} --ndvi --red 0 --nir 3 --divide-by 255
