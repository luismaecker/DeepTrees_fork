#!/bin/bash

#SBATCH --job-name=download_job
#SBATCH --output=download_job.log
#SBATCH --time=00:05:00  # 5 minutes time limit

echo "Hello from the SLURM job!"
date

# Add your download commands here, for example:
# wget http://example.com/data.txt
