#!/bin/bash -l
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --mpi=none 
#SBATCH --time=0-03:00:00
#SBATCH -p batch
#SBATCH --qos=qos-batch
#SBATCH -C skylake
#SBATCH -J FeatureModel2Keras
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=salah.ghamizi@uni.lu

echo "Hello from the batch queue on node ${SLURM_NODELIST} for neural architecture generation"
module purge
deactivate
module load math/Keras/2.1.6-foss-2018a-TensorFlow-1.8.0-Python-3.6.4

#srun --mpi=none -p gpu -n 1 --gres=gpu:1 --time=5:0 --pty bash -i                                                  
module load lib/TensorFlow/1.8.0-foss-2018a-Python-3.6.4-CUDA-9.1.85
pip install --upgrade --user  keras

python full.py
# Your more useful application can be started below!
#  dos2unix job.sh sbatch job.sh