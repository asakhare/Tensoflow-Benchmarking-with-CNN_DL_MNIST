!#/bin/bash

pwd
module load tensorflow/1.7_py2_gpu
source activate
echo "Executing Worker : 1 on host : $(hostname) "
python $SCRATCH/DistTF/distributed_deep_mnist_temp.py --batch_size=128 --job_name='worker' --task_index=2 --ps_hosts='gpu017.pvt.bridges.psc.edu:2222' --worker_hosts='gpu018.pvt.bridges.psc.edu:2222,gpu019.pvt.bridges.psc.edu:2222' > worker_1_${hostname}.out
