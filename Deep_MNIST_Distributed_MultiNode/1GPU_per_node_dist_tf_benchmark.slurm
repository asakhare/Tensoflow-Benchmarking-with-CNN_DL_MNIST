#!/bin/bash
#SBATCH --job-name=1GPUPerNode_DeepMNIST
#SBATCH --partition=GPU
#SBATCH -N 3
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks-per-node=2
##SBATCH --reservation=anirban2
#SBATCH --time=00:15:00

env

#cd $SCRATCH
#mkdir -p distTF.$SLURM_JOBID
#cd distTF.$SLURM_JOBID

echo $pwd

echo "Executing shell_Deep_Mnist_1GPU_Per_Node.sh"

srun $SCRATCH/DistTF/shell_Deep_Mnist_1GPU_Per_Node.sh
