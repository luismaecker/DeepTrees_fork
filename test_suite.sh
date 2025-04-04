#!/bin/sh
#SBATCH -A ka1176
#SBATCH -p vader 
#SBATCH --time=01:00:00
#SBATCH --exclusive

# These scripts need to run so that we can be sure that training, fine-tuning, inference, and averaging inference work

source ~/.bashrc
conda activate deeptree

# Run scripts as Python modules
python -m scripts.train trainer.fast_dev_run=True output_dir=test_suite
python -m scripts.train --config-name=finetune_halle trainer.fast_dev_run=True data.ground_truth_config.labels=null output_dir=test_suite
python -m scripts.test data.ground_truth_config.labels=null +trainer.fast_dev_run=True output_dir=test_suite
python -m scripts.test --config-name=inference_labeled_pretrained_halle data.ground_truth_config.labels=null +trainer.fast_dev_run=True output_dir=test_suite
python -m deeptrees.inference --config_path=./config/predict/inference_on_individual_tiles.yaml --image_path=/work/ka1176/shared_data/2024-ufz-deeptree/polygon-labelling/pool_tiles/tile_10_3.tif
