#!/bin/bash
#SBATCH --job-name=NLMCXR-convirt-vit-32
#SBATCH --array=0-1  # indices 0, 1, and 2
#SBATCH --output=logs/sbatch/NLMCXR-convirt-vit-32-%A_%a.out
#SBATCH --error=logs/sbatch/NLMCXR-convirt-vit-32-%A_%a.err
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

# Define experiment paths
experiments=(
  "finetune/retrieval/convirt/vit_32/finetune_nlmcxr_subset_01_convirt_vit_32"
  "finetune/retrieval/convirt/vit_32/finetune_nlmcxr_subset_10_convirt_vit_32"
)

# Get the current experiment based on the Slurm array index
experiment=${experiments[$SLURM_ARRAY_TASK_ID]}

# Run the program
# Use srun to run on a slurm node
srun python src/finetune_nlmcxr.py --config-name=finetune_nlmcxr_vit +experiment=$experiment