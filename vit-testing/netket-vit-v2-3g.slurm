#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:3g.40gb:1

#SBATCH --output=gpu_job-%j.out
#SBATCH --error=gpu_job-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=10G
#SBATCH --time=06:00:00

# to see ID and state of GPUs assigned
nvidia-smi

## Load the relevant modules needed for the job
module load gnu/12.3.0
module load python/3.12.1-33

## Source the venv
source /home/tcosgrov/code/netket-testing/.venv/bin/activate

#Execute
python /home/tcosgrov/code/netket-testing/vit-testing/netket-vit.py