#!/bin/bash
#SBATCH --job-name=test-hello-world
#SBATCH --output=logs/sbatch/test-hello-world-%A.out
#SBATCH --error=logs/sbatch/test-hello-world-%A.err
#SBATCH --mail-user=go34ded@mytum.de
#SBATCH --mail-type=ALL
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=0-00:01:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G


# change directory to the one containing the script
cd /vol/aimspace/users/$USER/vlp

# Load python module
ml python/anaconda3
# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that. Not necessary if you
# always run this script from a clean terminal
conda deactivate
conda deactivate
# If the following does not work, try 'source activate <env-name>'
conda activate convirt_fed
# Cache data to local /tmp directory (optional)
#rsync -r /vol/aimspace/projects/<dataset> /tmp
# Run the program
python -c "print('Hello World!')"