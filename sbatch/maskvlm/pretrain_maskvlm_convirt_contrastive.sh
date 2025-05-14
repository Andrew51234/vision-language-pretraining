#!/bin/bash
#SBATCH --job-name=maskvlm_convirt_contrastive
#SBATCH --output=logs/sbatch/maskvlm_convirt_contrastive-%A.out
#SBATCH --error=logs/sbatch/maskvlm_convirt_contrastive-%A.err
#SBATCH --mail-user=go34ded@mytum.de
#SBATCH --mail-type=ALL
##SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=3-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G


# change directory to the one containing the script
cd /vol/aimspace/users/$USER/vlp

# Load python module
ml python/anaconda3
# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that. Not necessary if you
# always run this script from a clean terminal
conda deactivate

# If the following does not work, try 'source activate <env-name>'
conda activate convirt_fed

# Run the program
# Use srun to run on a slurm node
srun python src/pretrain.py +experiment=maskvlm_convirt_contrastive