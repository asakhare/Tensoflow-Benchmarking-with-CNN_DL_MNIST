import tensorflow as tf
import time
import os
import re
import sys

#task_name = sys.argv[1]
#print("Task Name = " + str(task_name))

#Make Sure it sees the devices
from tensorflow.python.client import device_lib
print("*******Device Info Seen By TensorFlow*******")
print(device_lib.list_local_devices())
print("********************************************")

def create_nodelist(SLURM_JOB_NODELIST):
	nodes = re.findall(r'\d+', SLURM_JOB_NODELIST)
	nodelist = [i + '.pvt.bridges.psc.edu:2222' for i in nodes]
	return nodelist

def check_available_gpus():
	local_devices = device_lib.list_local_devices()
	gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
	gpu_num = len(gpu_names)

	print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))

	return gpu_num

def get_nodelist():
	node_count = int(os.environ['SLURM_JOB_NUM_NODES'])
	SLURM_JOB_NODELIST = os.environ['SLURM_JOB_NODELIST']
	print('node count=' + str(node_count))
	print('SLURM_JOB_NODELIST='+SLURM_JOB_NODELIST)
	if node_count >1:
		nodelist = create_nodelist(SLURM_JOB_NODELIST)
		print(nodelist)
	else:
		nodelist = [SLURM_JOB_NODELIST + '.pvt.bridges.psc.edu:2222']
	return nodelist

def create_cluster_dict(nodelist):
	d = {}
	ps_list = []
	worker_list = []
	for i in range(len(nodelist)):
		if i == 0:
			ps_list.append(nodelist[i])
		else:
			worker_list.append(nodelist[i])
	d['ps'] = ps_list
	d['worker'] = worker_list
	return d

nodelist = get_nodelist()
print('Nodelist::')
print(nodelist)

d = create_cluster_dict(nodelist)
print("Cluster Spec :: ")
print(d)

cluster = tf.train.ClusterSpec(d)

print("Cluster Created :: ")
cluster.as_dict()

#server = tf.train.Server(cluster, job_name="worker", task_index=0)
with tf.device("/job:ps/task:0/gpu:0"):
	a = tf.Variable(tf.truncated_normal(shape=[2]),dtype=tf.float32)
	print("a done")
with tf.device("/job:ps/task:0/gpu:1"):
	b = tf.Variable(tf.truncated_normal(shape=[2]),dtype=tf.float32)
	print("b done")
with tf.device("/job:worker/task:0/gpu:0"):
	c = tf.Variable(tf.truncated_normal(shape=[2]),dtype=tf.float32)
	print("c done")
#with tf.device("/job:ps/task:0/gpu:0"):
	ans = tf.Variable(tf.truncated_normal(shape=[2]),dtype=tf.float32)
	ans = a + b + c 
	print("addition done")
	#with tf.device("/job:ps/task:0/gpu:1"):
	target = tf.constant(100.,shape=[2],dtype=tf.float32)
	loss = tf.reduce_mean(tf.square(ans-target))
	opt = tf.train.GradientDescentOptimizer(.0001).minimize(loss)
	print("optimizer done")
#with tf.device("/job:ps/task:0/cpu:0"):
#	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	#sess = tf.train.MonitoredTrainingSession(master=server.target,config=tf.ConfigProto(log_device_placement=True))
	#tar = "grpc://" + str(d['ps'][0])
with tf.device("/job:ps/task:0/gpu:0"):
	with tf.Session() as sess:
		print("Entered Session")
		for i in range(100):
			print("Entered Loop")
			#if sess.should_stop(): break
			sess.run(opt)
			if i % 10 == 0:
				r = sess.run(c)
				print(r)
		sess.close()
	
print("Finish")
#WIP
