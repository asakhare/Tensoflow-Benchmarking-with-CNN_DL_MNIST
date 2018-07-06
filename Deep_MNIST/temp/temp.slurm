#!/bin/bash
#SBATCH --job-name=Variable_Test
echo "Batch Size"
##echo "without dollar"
##echo batch_size
##python temp.py batch_size
echo "with dollar"
echo $batch_size
python temp.py $batch_size
echo "Done"
