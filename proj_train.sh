#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=14:00:00
#SBATCH --output=train.txt
#SBATCH --gres=gpu:rtx8000:4

TRAINDIR='./data/DIV2K_train_HR/'
VALIDDIR='./data/DIV2K_valid_HR/'

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
module load cuda/11.3.1
conda create --name venv

python train.py --upscale_factor 4 --cuda

conda deactivate
