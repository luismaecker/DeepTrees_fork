#!/bin/sh
#SBATCH -A ka1176
#SBATCH -p vader 
#SBATCH --time=01:00:00
#SBATCH --exclusive

# These scripts need to run so that we can be sure that training, fine-tuning, inference, and averaging inference work

source ~/.bashrc
conda activate /work/ka1176/harsh/conda_envs/deeptree


# python scripts/train.py trainer.fast_dev_run=True output_dir=test_suite
# python scripts/train.py --config-name=finetune_halle trainer.fast_dev_run=True data.ground_truth_config.labels=null output_dir=test_suite
# python scripts/test.py data.ground_truth_config.labels=null +trainer.fast_dev_run=True output_dir=test_suite
# python scripts/test.py --config-name=inference_labeled_pretrained_halle data.ground_truth_config.labels=null +trainer.fast_dev_run=True output_dir=test_suite
python deeptrees/inference.py --config_name=inference_on_individual_tiles --image_path=/work/ka1176/shared_data/2024-ufz-deeptree/polygon-labelling/pool_tiles/tile_10_3.tif 