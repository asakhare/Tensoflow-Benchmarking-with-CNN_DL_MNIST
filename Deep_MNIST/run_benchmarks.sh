#!/bin/bash
#rundir=2018-05-31/0cores_1-K80
#mkdir -p $rundir
#pushd $rundir
#  cp 
#  sbatch -p GPU --gres=gpu:k80:1 
#popd

##Clear the output directory
cd ./results/
mv *:* ./archived/
cd ..

rootdir = `pwd`'/results/'

##t=`date`;
t=`date "+%Y-%m-%d"`
##t=2018-05-31;
##ofile="$t-k80=1";sbatch -p GPU-shared --gres=gpu:k80:1 -o $ofile gpuJob.slurm
##ofile="$t-k80=2";sbatch -p GPU-shared --gres=gpu:k80:2 -o $ofile gpuJob.slurm

for num_gpu in 1 2 4; do
   for gpu_type in k80 p100; do
   	echo "**********************************************************"
   	if [ $num_gpu == 4 ]; then 
   		if [ $gpu_type == "k80" ]; then
   			echo "     NUM_GPU = $num_gpu"
  			echo "     GPU_TYPE = $gpu_type"
   			echo "     Date = $t" 
   			ofile="$t-$gpu_type:$num_gpu:scratch";sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile gpuJob.slurm
			ofile="$t-$gpu_type:$num_gpu:local";sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile gpuJob_local.slurm
   			echo "----------------------------------------------------------"
   		else
   			continue
   		fi
   	else
   		echo "     NUM_GPU: = $num_gpu"
   		echo "     GPU_TYPE = $gpu_type"
   		echo "     Date = $t"
   		ofile="$t-$gpu_type:$num_gpu:scratch";sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile gpuJob.slurm
		ofile="$t-$gpu_type:$num_gpu:local";sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile gpuJob_local.slurm
   		echo "----------------------------------------------------------"
   	fi
   done 
done

##echo 'Creating CSV File at'
##echo $rootdir

##python tf_benchmarking_file_creation.py $rootdir


