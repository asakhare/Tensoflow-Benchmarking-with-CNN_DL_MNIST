#!/bin/bash
#SBATCH --job-name=tf_CNN_MNIST_nogpu
##SBATCH --partition=GPU-shared
##SBATCH --nodes=1
##SBATCH --gres=gpu:p100:1
#SBATCH --time=03:00:00

set -x
date 

echo "Set up environment"
set +x; module load tensorflow/1.7_py2_nogpu; set -x
source activate

cd $LOCAL
cp /pylon5/cc5fp8p/asakhare/tf_benchmarks/Deep_MNIST/CNN_MNIST.py /pylon5/cc5fp8p/asakhare/tf_benchmarks/Deep_MNIST/t*.gz .

echo "Log some information"
set +x; module list; set -x
lscpu | grep -i 'model name'
scontrol show jobid=$SLURM_JOB_ID
env
which python

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

echo "Run"
echo "batch_size=$batch_size"
date
time python CNN_MNIST_CPU_ONLY.py $batch_size
date
