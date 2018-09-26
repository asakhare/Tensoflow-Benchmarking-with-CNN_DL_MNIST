from __future__ import print_function
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import os
import numpy as np
import re
import sys
import datetime
import time

FLAGS = None

def print_visible_local_devices():
        from tensorflow.python.client import device_lib
        print("*******Device Info Seen By TensorFlow*******")
        print(device_lib.list_local_devices())
        print("********************************************")

def create_nodelist(SLURM_JOB_NODELIST):
        nodes = re.findall(r'\d+', SLURM_JOB_NODELIST)
        nodelist = []
        new_nodes = []
        if '-' in SLURM_JOB_NODELIST:
                if ',' in SLURM_JOB_NODELIST:
                        cont_node_list = re.findall(r'\d+-\d+',SLURM_JOB_NODELIST)
                        nodes = ['gpu' + str(i) for i in re.findall(r'\d+', SLURM_JOB_NODELIST)]
                        new_nodes=[]
                        for i in cont_node_list:
                                temp=[int(i) for i in(re.findall(r'\d+',i))]
                                for i in range(temp[0],temp[1]+1):
                                        new_nodes.append('gpu0' + str(i))
                        for i in nodes:
                                if i not in new_nodes:
                                        new_nodes.append(i)
                        new_nodes=sorted(new_nodes)
                else:
                        new_nodes=[]
                        node_int=[int(i) for i in nodes]
                        for i in range(node_int[0],node_int[1]+1):
                                new_nodes.append('gpu0' + str(i))
        else:
                new_nodes=['gpu'+ str(i) for i in nodes]
        nodelist = [i + '.pvt.bridges.psc.edu:2222' for i in new_nodes]
        return nodelist

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
                        #temp = nodelist[i].replace('2222','2223')
                        #worker_list.append(temp)
                else:
                        worker_list.append(nodelist[i])
        d['ps'] = ps_list
        d['worker'] = worker_list
        return d


def check_available_gpus():
        from tensorflow.python.client import device_lib
        local_devices = device_lib.list_local_devices()
        gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
        gpu_num = len(gpu_names)
        print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))
        return gpu_num


def model_forward(x_image,keep_prob):
        with tf.variable_scope('conv1') as scope:
                with tf.device("/job:ps/task:0"):
                        W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
                        b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]))
                h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
                h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('conv2') as scope:
                with tf.device("/job:ps/task:0"):
                        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
                        b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]))
                h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
                h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('dense1') as scope:
                with tf.device("/job:ps/task:0"):
                        W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
                        b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]))
                h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        with tf.variable_scope('dense2') as scope:
                with tf.device("/job:ps/task:0"):
                        W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
                        b_fc2 = tf.Variable(tf.constant(0.1,shape=[10]))
                #keep_prob = tf.placeholder(tf.float32)
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
                y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv

# Flags for defining the tf.train.ClusterSpec
#tf.app.flags.DEFINE_string("ps_hosts", "",
#                           "Comma-separated list of hostname:port pairs")
#tf.app.flags.DEFINE_string("worker_hosts", "",
#                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("hidden_units", 100,
                            "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_string("data_dir", "/tmp/tensorflow/mnist/input_data/",
                           "Directory for storing mnist data")
tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")
tf.app.flags.DEFINE_integer("max_step", 100, "Training batch size")

FLAGS = tf.app.flags.FLAGS

IMAGE_PIXELS = 28

def main(_):
	nodelist = get_nodelist()
	print('Nodelist::')
	print(nodelist)
	d = create_cluster_dict(nodelist)
	print("Cluster Spec :: ")
	print(d)
	cluster = tf.train.ClusterSpec(d)
	print("Cluster Created :: ")
	cluster.as_dict()
	print_visible_local_devices()
	num_worker_machines = int(os.environ['SLURM_JOB_NUM_NODES']) - 1
	num_workers = check_available_gpus()* num_worker_machines
	num_gpu = check_available_gpus()
	print("Total Number of Workers = " + str(num_workers))

	#ps_hosts = FLAGS.ps_hosts.split(",")
	#worker_hosts = FLAGS.worker_hosts.split(",")

	# Create a cluster from the parameter server and worker hosts.
	#cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

	# Create and start a server for the local task.
	#server = tf.train.Server(cluster,
	#                         job_name=FLAGS.job_name,
	#                         task_index=FLAGS.task_index)

	if FLAGS.job_name == "ps":
		server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
		print("Parameter Server Starting")
		server.join()
	elif FLAGS.job_name == "worker":
		FLAGS.task_index=FLAGS.task_index-1
		print("Task Name::" + str(FLAGS.job_name))
		print("Task Index::" + str(FLAGS.task_index))
		server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

		# Assigns ops to the local worker by default.
		for i in range(num_gpu):	
			with tf.device(tf.train.replica_device_setter(worker_device=('/job:worker/task:{0}/gpu:{1}').format(FLAGS.task_index,i),cluster=cluster)):
				# Variables of the hidden layer
				print("GPU Number::" + str(i))
				hid_w = tf.Variable(
						tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
						stddev=1.0 / IMAGE_PIXELS), name="hid_w")
				hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")
				run_metadata = tf.RunMetadata()
				# Variables of the softmax layer
				sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],stddev=1.0 / math.sqrt(FLAGS.hidden_units)),name="sm_w")
				sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

				x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
				y_ = tf.placeholder(tf.float32, [None, 10])

				hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
				hid = tf.nn.relu(hid_lin)

				y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
				loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

				global_step = tf.Variable(0)

				train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

				correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

				#saver = tf.train.Saver()
				summary_op = tf.summary.merge_all()
				init_op = tf.initialize_all_variables()

		# Create a "supervisor", which oversees the training process.
		sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="./logs_%d" % FLAGS.task_index,
                             init_op=init_op,
                             summary_op=summary_op,
                             #saver=saver,
                             global_step=global_step,
                             #save_model_secs=60
				)

		mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

		begin_time = time.time()
		frequency = 100
    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
		print("Target Server :: " + str(server.target))
		with sv.managed_session(server.target,config=tf.ConfigProto(allow_soft_placement = True,log_device_placement=True)) as sess:
      # Loop until the supervisor shuts down or 100000 steps have completed.
			step = 0
      #sess.run(tf.global_variables_initializer(),options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
			while not sv.should_stop() and step < 100000:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
				start_time = time.time()

				batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
				train_feed = {x: batch_xs, y_: batch_ys}

				_, step = sess.run([train_op, global_step], feed_dict=train_feed)
				elapsed_time = time.time() - start_time
				if step % frequency == 0: 
					print ("Done step %d" % step, " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))


      # Test trained model
			print("Test-Accuracy: %2.4f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

		print("Total Time: %3.2fs" % float(time.time() - begin_time))
    # Ask for all the services to stop.
		sv.stop()
    #for device in run_metadata.step_stats.dev_stats:
    	#print(device.device)
    	#for node in device.node_stats:
        	#print("  ", node.node_name)


if __name__ == "__main__":
  tf.app.run()
