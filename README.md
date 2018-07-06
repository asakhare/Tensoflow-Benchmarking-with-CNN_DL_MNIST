# Tensoflow-Benchmarking-with-CNN_DL_MNIST

## P01. DL-Benchmarks-TensorFlow - Phase 1

### Project Description
Execution of DL experiments on TensorFlow with different deployment architectures to understand relation between DL training speed and hardware. The first phase includes execution of below given experiments (Experiment ID – 0 to 9). 

		TensorFlow		TensorFlow	Slurm		
    
| ExpID | GPUType | #GPUs | CPU Type | #CPUs | #Cores |	Quiet? | Staged from? |
|-------|---------|-------|----------|-------|--------|--------|--------------|
|0 | K80 |	1	| E5-2695 v3 | - | - | No	| $LOCAL |
|1 | K80 | 2 |	E5-2695 v3 | - | - | No	| $LOCAL |
|2 | K80 |	4 |	E5-2695 v3 | - | - | No |	$LOCAL |
|3 |	P100	| 1 |	E5-2683 v4	| - |	- |	No |	$LOCAL |
|4 |	P100	| 2	| E5-2683 v4	| - |	- |	No |	$LOCAL |
|5 |	K80 |	1	| E5-2695 v3 |	- |	- |	No | $SCRATCH |
|6 |	K80	| 2	| E5-2695 v3 |	- |	-	| No	| $SCRATCH |
|7 |	K80	| 4	| E5-2695 v3	| - |	- |	No |	$SCRATCH |
|8 |	P100	| 1	| E5-2683 v4	| - |	-	| No	| $SCRATCH |
|9 |	P100	| 2	| E5-2683 v4	| -	| -	| No	| $SCRATCH |

### DL Models:
	Convolutional model: DeepMNIST (https://www.tensorflow.org/versions/r1.2/get_started/mnist/pros)
	Dataset: MNIST
	Directories: $SCRATCH and $LOCAL, GPU TYPES: k80 & p100, # of GPUs: 1,2,4


### Scripts:

#### CNN_MNIST.py
This is the file which stores the python code for training a convolutional neural network (CNN) on the MNIST dataset. It is primarily taken from the TensorFlow tutorial with small modifications. In particular, the deprecated tf.nn.softmax_cross_entropy_with_logits(...) is 
replaced with tf.nn.softmax_cross_entropy_with_logits_v2(...). Along with that, some header code to print local devices from TensorFlow (line 13-15), code to record training time, and to calculate the training speed is also added.

The performance reporting is done on the basis of the following parameters:
	Mean Time: Average training time per step. This parameter gives an idea about the average training time required for the model with the given batch size.

	Mean Speed (Images/Sec): Mean speed is the training speed of the convolutional deep learning model in images/sec. 
   Formula: Batch Size / Mean Time

	Speed Uncertainty: +/- Variation in Mean Speed
  Formula: Standard Deviation of Speed For distribution across steps / √Number of steps

	Speed Jitter: This parameter gives an idea about the spread of the speeds across multiple steps.
	* Median [ k=0nSpeed k – MeadianSpeed ]	

#### gpuJob.slurm

This is a batch file which holds the code to run the CNN_MNIST.py file in $SCRATCH directory. This file takes care of setting up the tensorflow environment, activating the environment, logging information to the output file, and running the CNN_MNIST.py script. This file is called from run_benchmarks.sh script.

To run the file separately use sbatch command. (please refer: https://www.psc.edu/bridges/user-guide/running-jobs)

#### gpuJob_local.slurm
This is a batch file which holds the code to run the CNN_MNIST.py file in $LOCAL directory. This file takes care of setting up the tensorflow environment, activating the environment, logging information to the output file, and running the CNN_MNIST.py script.

To run the file separately use sbatch command. (please refer: https://www.psc.edu/bridges/user-guide/running-jobs)

#### run_benchmarks.sh
This is the main file which needs to be run for executing the TensorFlow Deep MNIST benchmarking experiments. Brief flow of the shell script.

This file stores the output files for each of the outputs of the model execution of each deployment architecture, in ‘results’ directory which should be a sub directory of the current directory.  
First it moves any older files from results directory to archived directory which is a subdirectory of the results directory.
Then it enters a nested loop and submits the required jobs to the job queue (Exp ID 0 to 9).
All the output files are stored in the result directory and follow the following filename format. YYYY-MM-DD-<gpu type>:<number of gpus>:<staged from directory>
e.g. An experiment run on 4th Jun 2018 on a k80 GPU from local directory will have an output file name as ‘2018-06-04-k80:1:local’. 
Command to run this file is : ‘ . run_benchmarks.sh’ 

#### tf_benchmarking_file_creation.py
This script scans all the output files and stores the important outputs metrices in a comma delimited csv file. A sample file created by this script is as given below.

|mean_speed	| Mean_Time_Per_Step |	speed_uncertainty |	Speed_Jitter |	gpu_type |	num_gpu	|	staged_from |
|-----------|--------------------|--------------------|--------------|-----------|----------|-------------|
|0 |	8732.961728 |	0.005725434	|	6.388218371	|	760.2575007	| p100	|	2	|	$SCRATCH |
|1|	5250.901566|	0.009522174	|	0.734560314	|	65.77123005	| k80	|	2	|	$SCRATCH
|2|	4836.64568	|0.010337743	|	2.079079801	|	203.1963323 |	k80	|	4	|	$LOCAL
|3|	5195.149557	|0.009624362	|	0.750577503	|	62.74329595	| k80	|	4	|	$SCRATCH
|4|	5226.787128	|0.009566106	|	0.701725294	|	70.62155769	 |k80	|	2	|	$LOCAL
|5|	12247.0821	|0.004082605	|	3.075517837	|	359.8248272	| p100	|	2	|	$LOCAL
|6|	9434.888805	|0.00529948	|	6.034101409	|	828.7332798	|p100	|	1|		$SCRATCH
|7|	2643.516179	|0.018914202	|	3.94097403	|	340.0571688|	k80	|	1	|	$LOCAL
|8|	9600.754615	|0.005207924	|	5.879743298	|	762.022697	|p100	|	1	|	$LOCAL
|9|	2673.974139|	0.01869876	|	4.147340431	|	386.9301569	|k80	|	1	|	$SCRATCH


This script should be kept in the directory above the results directory. This script automatically scans through every file stored in results directory and fetches the above values from the file. An output file in the results directory is created which follows the following name format.
<today’s date>:TF_PERFORMANCE_RESULTS.csv
e.g. 2018-06-04:TF_PERFORMANCE_RESULTS.csv
Note: Since the submitted jobs take time to finish, the output files will be available in the results directory only after a job is completed. Hence before running this file make sure that all the jobs are completed to get the best results. Also, since this file uses Pandas module, make sure that Pandas is installed or run the file in AI_ENV. Commands to run the file in AI_ENV are given below.
$ module load AI/anaconda3-5.1.0_gpu
$ source activate $AI_ENV
$ python tf_benchmarking_file_creation.py

Note-Benchmarking Graphs With Various Deployment Architectures are in the Results Directory.

Authored By,
Anand Sakhare
Pittsburgh Supercomputing Center | Carnegie Mellon University
Email ID: asakhare@andrew.cmu.edu

