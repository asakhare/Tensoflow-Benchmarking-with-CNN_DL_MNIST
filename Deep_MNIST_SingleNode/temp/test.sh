#!/bin/bash

a="a_test"
b="b_test"
echo "shell"
echo $a
echo $b

c="sbatch --export=a=$a,b=$b,ALL -p RM -t 00:05:00"
echo $c

sbatch --export=a=$a,b=$b,c=$c,ALL -p RM -t 00:05:00 t.slurm

##sbatch -p RM --export=a=$a,b=$b test.slurm
