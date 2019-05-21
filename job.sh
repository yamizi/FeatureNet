#!/bin/bash -l

#SBATCH -n 10   
#SBATCH -N 4
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --time=2-23:00:00
#SBATCH --qos=qos-gpu
#SBATCH -C skylake
#SBATCH -J MNIST100-FeatureModel2Keras
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=salah.ghamizi@uni.lu

echo "Hello from the batch queue on node ${SLURM_NODELIST} for neural architecture generation"
module purge
module load math/Keras/2.1.6-foss-2018a-TensorFlow-1.8.0-Python-3.6.4

#srun --mpi=none -p gpu -n 1 --gres=gpu:1 --time=5:0 --pty bash -i                                                  
module load lib/TensorFlow/1.8.0-foss-2018a-Python-3.6.4-CUDA-9.1.85
pip install --upgrade --user  keras

python -u pledge_evolution.py -d 'mnist' -b '../products/run8' > bash_evol_pledge_mnist_100.out
# Your more useful application can be started below!
#  dos2unix job.sh sbatch job.sh
