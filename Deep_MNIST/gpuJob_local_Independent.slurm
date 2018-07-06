#!/bin/bash
#SBATCH --job-name=tf_CNN_MNIST
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --time=00:15:00

set -x
date 

echo "Set up environment"
set +x; module load tensorflow/1.7_py2_gpu; set -x
source activate

cd $LOCAL
cp /pylon5/cc5fp8p/asakhare/tf_benchmarks/Deep_MNIST/CNN_MNIST_INDEPENDENT.py /pylon5/cc5fp8p/asakhare/tf_benchmarks/Deep_MNIST/t*.gz .

echo "Log some information"
set +x; module list; set -x
scontrol show jobid=$SLURM_JOB_ID
env
which python

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

echo "Run"
date
time python CNN_MNIST_INDEPENDENT.py
date
