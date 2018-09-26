#!/bin/bash
#SBATCH --job-name=tf_CNN_MNIST
#SBATCH --partition=GPU-shared
#SBATCH --nodes=2
#SBATCH --gres=gpu:p100:1
#SBATCH --time=00:15:00


echo "Set up environment"
set +x; module load tensorflow/1.7_py2_gpu; set -x
source activate

echo "Log some information"
set +x; module list; set -x
scontrol show jobid=$SLURM_JOB_ID
env
which python

time python distributed-deep-mnist-with-queue.py --job_name=ps --data_dir="$SCRATCH/DistTF/" --log_dir="$SCRATCH/DistTF/logs" &

time python distributed-deep-mnist-with-queue.py --job_name=worker --task_index=0 --data_dir="$SCRATCH/DistTF/" --log_dir="$SCRATCH/DistTF/logs" &

time python distributed-deep-mnist-with-queue.py --job_name=worker --task_index=1 --data_dir="$SCRATCH/DistTF/" --log_dir="$SCRATCH/DistTF/logs" &
