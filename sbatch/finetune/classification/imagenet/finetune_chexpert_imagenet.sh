#!/bin/bash
#SBATCH --job-name=CheXpert-imagenet
#SBATCH --output=logs/sbatch/CheXpert-imagenet-%A.out
#SBATCH --error=logs/sbatch/CheXpert-imagenet-%A.err
#SBATCH --mail-user=viet.x.pham@tum.de
#SBATCH --mail-type=ALL
##SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=2-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G


# change directory to the one containing the script
cd /vol/aimspace/users/phvi/GitLab/vlp

# Load python module
ml python/anaconda3
# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that. Not necessary if you
# always run this script from a clean terminal

source deactivate

# If the following does not work, try 'source activate <env-name>'
source activate vlp-dev

# Run the program
# Use srun to run on a slurm node
srun python src/finetune_chexpert.py --config-name=finetune_chexpert_imagenet +experiment=finetune/classification/imagenet/finetune_chexpert_imagenet