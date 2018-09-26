#!/bin/bash

echo "SLURM"

echo "a"
echo $a

echo "b"
echo $b

echo "c"
echo $c

module load python 
python test.py $a $b
