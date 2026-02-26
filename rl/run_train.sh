#!/bin/bash
#SBATCH -o train_rl_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="rl_train"
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
##SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --account="research-ceg-tp"

echo "🚀 Starting RL Training Job"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"

# Load modules
module load miniconda3

# Setup conda
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate control

# Check environment
echo "🔍 Environment check:"
echo "  Python: $(which python)"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "  GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# Navigate to project directory
cd ${HOME}/Devs/control

# Run training
python marl/train_mappo.py

echo "✅ RL Training job completed at $(date)"
