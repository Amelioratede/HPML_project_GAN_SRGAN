#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=14:00:00
#SBATCH --output=test.txt
#SBATCH --gres=gpu:rtx8000:4

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
module load cuda/11.3.1
conda create --name venv

python test_image.py --image $IMG --weight $WEIGHT --cuda

conda deactivate
