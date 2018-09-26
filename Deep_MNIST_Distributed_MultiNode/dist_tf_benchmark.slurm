#!/bin/bash
#SBATCH --partition=GPU
#SBATCH -N 3
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks-per-node=1
##SBATCH --reservation=anirban
#SBATCH --time=00:15:00

env

#cd $SCRATCH
#mkdir -p distTF.$SLURM_JOBID
#cd distTF.$SLURM_JOBID

echo $pwd

srun $SCRATCH/DistTF/multi_node_seperation.sh
