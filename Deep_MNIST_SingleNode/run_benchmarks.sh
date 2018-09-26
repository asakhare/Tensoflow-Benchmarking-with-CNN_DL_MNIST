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

rootdir=`pwd`'/results/'

##t=`date`;
t=`date "+%Y-%m-%d"`
##t=2018-05-31;
##ofile="$t-k80=1";sbatch -p GPU-shared --gres=gpu:k80:1 -o $ofile gpuJob.slurm
##ofile="$t-k80=2";sbatch -p GPU-shared --gres=gpu:k80:2 -o $ofile gpuJob.slurm

##GPU Scenarios
'''
for num_gpu in 1 2 4; do
   for gpu_type in k80 p100; do
   	for batch_size in 16 32 64 128 256 512 1024 2048 4096 8192 16384 327681;do
		echo "**********************************************************"
   		if [ $num_gpu == 4 ]; then 
   			if [ $gpu_type == "k80" ]; then
				echo "     GPU Job"
   				echo "     NUM_GPU = $num_gpu"
  				echo "     GPU_TYPE = $gpu_type"
   				echo "     Date = $t" 
				echo "	   Batch_Size = $batch_size"
				commandstring="sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size\,ALL gpuJob.slurm"
   				ofile="$t-GPU-$gpu_type:$num_gpu:$batch_size:scratch:PS";sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size,commandstring="$commandstring",ALL gpuJob.slurm
				commandstring="sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size\,ALL gpuJob_local.slurm"
				ofile="$t-GPU-$gpu_type:$num_gpu:$batch_size:local:PS";sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size,commandstring="$commandstring",ALL gpuJob_local.slurm
				commandstring="sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size\,ALL gpuJob_Independent.slurm"	
				ofile="$t-GPU-$gpu_type:$num_gpu:$batch_size:scratch:IN";sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size,commandstring="$commandstring",ALL gpuJob_Independent.slurm
				commandstring="sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size\,ALL gpuJob_local_Independent.slurm"
				ofile="$t-GPU-$gpu_type:$num_gpu:$batch_size:local:IN";sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size,commandstring="$commandstring",ALL gpuJob_local_Independent.slurm
   				echo "----------------------------------------------------------"
   			else
   				continue
   			fi
   		else
			echo "     GPU Job"
   			echo "     NUM_GPU: = $num_gpu"
   			echo "     GPU_TYPE = $gpu_type"
   			echo "     Date = $t"
			echo "     Batch_Size = $batch_size"
			commandstring="sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size\,ALL gpuJob.slurm"
			ofile="$t-GPU-$gpu_type:$num_gpu:$batch_size:scratch:PS";sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size,commandstring="$commandstring",ALL gpuJob.slurm
			commandstring="sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size\,ALL gpuJob_local.slurm"
			ofile="$t-GPU-$gpu_type:$num_gpu:$batch_size:local:PS";sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size,commandstring="$commandstring",ALL gpuJob_local.slurm			
			commandstring="sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size\,ALL gpuJob_Independent.slurm"
			ofile="$t-GPU-$gpu_type:$num_gpu:$batch_size:scratch:IN";sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size,commandstring="$commandstring",ALL gpuJob_Independent.slurm
			commandstring="sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size\,ALL gpuJob_local_Independent.slurm"
			ofile="$t-GPU-$gpu_type:$num_gpu:$batch_size:local:IN";sbatch -p GPU-shared --gres=gpu:$gpu_type:$num_gpu -o ./results/$ofile --export=batch_size=$batch_size,commandstring="$commandstring",ALL gpuJob_local_Independent.slurm
   			echo "----------------------------------------------------------"
   		fi
	done
   done 
done
'''

##CPU Only Scenarios
for num_cores in 1 2 4 7 8 14 16 28 32; do
	for partition in RM GPU;do
		for batch_size in 16 32 64 128 256 512 1024 2048 4096 8192 16384;do
			echo "**********************************************************"
			if [ $num_cores == 32 -a $partition == "RM" ]; then
				continue
			else
				echo "     CPU Only Job"
				echo "     NUM_CORES = $num_cores"
				echo "     PARTITION = $partition"
				echo "     Date = $t"
				echo "	   Batch Size = $batch_size"
				if [ $partition == "RM" ]; then	
					commandstring="sbatch -p $partition -N 1 --ntasks-per-node=$num_cores -o ./results/$ofile --export=batch_size=$batch_size\,ALL cpuJob.slurm"
					ofile="$t-CPU-$partition:E5-2695-V2:$num_cores:$batch_size:scratch";sbatch -p $partition -N 1 --ntasks-per-node=$num_cores -o ./results/$ofile --export=batch_size=$batch_size,commandstring="$commandstring",ALL cpuJob.slurm
					commandstring="sbatch -p $partition -N 1 --ntasks-per-node=$num_cores -o ./results/$ofile --export=batch_size=$batch_size\,ALL cpuJob_local.slurm"
					ofile="$t-CPU-$partition:E5-2695-V2:$num_cores:$batch_size:local";sbatch -p $partition -N 1 --ntasks-per-node=$num_cores -o ./results/$ofile --export=batch_size=$batch_size,commandstring="$commandstring",ALL cpuJob_local.slurm
				else
					commandstring="sbatch -p $partition -N 1 --ntasks-per-node=$num_cores --gres=gpu:p100:2 -o ./results/$ofile --export=batch_size=$batch_size\,ALL cpuJob.slurm"
					ofile="$t-CPU-$partition:E5-2683-V4:$num_cores:$batch_size:scratch";sbatch -p $partition -N 1 --ntasks-per-node=$num_cores --gres=gpu:p100:2 -o ./results/$ofile --export=batch_size=$batch_size,commandstring="$commandstring",ALL cpuJob.slurm
					commandstring="sbatch -p $partition -N 1 --ntasks-per-node=$num_cores --gres=gpu:p100:2 -o ./results/$ofile --export=batch_size=$batch_size\,ALL cpuJob_local.slurm"
					ofile="$t-CPU-$partition:E5-2683-V4:$num_cores:$batch_size:local";sbatch -p $partition -N 1 --ntasks-per-node=$num_cores --gres=gpu:p100:2 -o ./results/$ofile --export=batch_size=$batch_size,commandstring="$commandstring",ALL cpuJob_local.slurm
				fi
			fi
			echo "----------------------------------------------------------"
		done
	done
done

## XLM Node scenarios - CPU E78880 V4
'''
for num_cores in 1 2 4 7 8 14 16 28 32 36 72 144 288;do
	for batch_size in 16 32 64 128 256 512 1024 2048 4096 8192 16384;do
		echo "**********************************************************"	
		echo "     CPU Only Job"
		echo "     NUM_CORES = $num_cores"
		echo "     PARTITION = XML"
		echo "     Date = $t"
		echo "	   Batch Size=$batch_size"
		ofile="$t-CPU-XLM:E7-8880-V4:$num_cores:local";sbatch -p XLM --mem=12000GB -C PH2 --ntasks-per-node=$num_cores -o ./results/$ofile --export=batch_size=$batch_size cpuJob_local.slurm
		echo "----------------------------------------------------------"	
	done
done
'''
##echo 'Creating CSV File at'
##echo $rootdir

##python tf_benchmarking_file_creation.py $rootdir

