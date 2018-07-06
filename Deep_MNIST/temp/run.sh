#!/bin/bash
batch_size='32'
sbatch -p GPU-shared --gres=gpu:p100:1 --export=batch_size=$batch_size temp.slurm
batch_size='45'
sbatch -p RM --export=batch_size=$batch_size temp.slurm

