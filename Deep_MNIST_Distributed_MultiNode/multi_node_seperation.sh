#!/bin/sh

cd $SCRATCH/DistTF/

echo "**********************Environment Variables**********************"
env ##> ${SLURM_NODEID}_${SLURM_PROCID}.env 
echo "*****************************************************************"
num_nodes=$SLURM_JOB_NUM_NODES
echo $SLURM_NODEID

node_id=$SLURM_NODEID

a=$SLURM_JOB_GPUS
b=${a//[^[:digit:]]/}
num_gpu="${#b}"

num_gpu=$((num_gpu - 1))

module load tensorflow/1.7_py2_gpu
source activate

if [ $node_id -eq 0 ]; then
	echo "Executing PS on  Node : $SLURM_NODEID on host : $(hostname) "
	python distributed_deep_mnist.py --batch_size=128 --job_name='ps' --task_index=0 --data_dir='.' >& ps_0_${hostname}.out
else
	echo "Executing Worker on Node : $SLURM_NODEID on host : $(hostname) "
	python distributed_deep_mnist.py --batch_size=128 --job_name='worker' --task_index=$node_id --data_dir='.' >& worker_${node_id}_${hostname}.out
fi
