#!/bin/bash
#SBATCH --job-name=deep_mnist_distributed
#SBATCH --partition=GPU-shared
#SBATCH --nodes=2
##SBATCH --gres=gpu:p100:2
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

batch_size=64

echo "Run"
echo "batch_size=$batch_size"
date
time python deep_mnist_dist.py $batch_size
date

