#!/bin/bash
#SBATCH --job-name=tf_CNN_MNIST
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
##SBATCH --gres=gpu:p100:1
#SBATCH --time=00:15:00

set -x
date 

echo "Original Input Command"
echo $commandstring

echo "Set up environment"
set +x; module load tensorflow/1.7_py2_gpu; set -x
source activate

echo "Log some information"
set +x; module list; set -x
scontrol show jobid=$SLURM_JOB_ID
env
which python

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

echo "Run"
echo "batch_size=$batch_size"
date
time python CNN_MNIST_INDEPENDENT.py $batch_size
date
