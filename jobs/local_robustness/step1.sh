#!/bin/bash -l

#SBATCH -n 2
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --time=0-01:00:00
#SBATCH --qos=qos-gpu
#SBATCH -C skylake
#SBATCH -J Step1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=salah.ghamizi@uni.lu

echo "Hello from the batch queue on node ${SLURM_NODELIST} for neural architecture mutation"
module purge
module load swenv/default-env/v1.1-20180716-production lang/Python/3.6.4-foss-2018a system/CUDA numlib/cuDNN math/Gurobi/8.1.1-intel-2018a-Python-3.6.4
pip install --user -r ../requirements.txt

python -u experiments/local_robustness/step1.py > step1.out
