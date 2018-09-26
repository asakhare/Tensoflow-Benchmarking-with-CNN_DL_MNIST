#!/bin/bash
#SBATCH --job-name=tf_CNN_MNIST
#SBATCH --partition=GPU-shared
##SBATCH --nodes=1
##SBATCH --gres=gpu:p100:1
#SBATCH --time=02:15:00

set -x
date 

echo "Set up environment"
set +x; module load tensorflow/1.5_gpu; set -x
source activate

echo "Log some information"
set +x; module list; set -x
scontrol show jobid=$SLURM_JOB_ID
env
which python

echo "Run"
date
time python CNN_MNIST_CPU_ONLY.py
date
