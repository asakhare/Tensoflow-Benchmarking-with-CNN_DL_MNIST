from __future__ import print_function
import tensorflow as tf
import argparse
import os
import numpy as np
import re
import sys
import datetime
import time
from tensorflow.examples.tutorials.mnist import input_data


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
        nodelist = [i + '.pvt.bridges.psc.edu:2223' for i in new_nodes]
        return nodelist

def check_available_gpus():
        from tensorflow.python.client import device_lib
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
                nodelist = [SLURM_JOB_NODELIST + '.pvt.bridges.psc.edu:2223']
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

def model_forward(x_image,keep_prob):
        with tf.variable_scope('conv1') as scope:
                ##with tf.device("/job:ps/task:0"):
                W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
                b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]))
                h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
                h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('conv2') as scope:
                ##with tf.device("/job:ps/task:0"):
                W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
                b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]))
                h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
                h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('dense1') as scope:
                ##with tf.device("/job:ps/task:0"):
                W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
                b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]))
                h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        with tf.variable_scope('dense2') as scope:
                ##with tf.device("/job:ps/task:0"):
                W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
                b_fc2 = tf.Variable(tf.constant(0.1,shape=[10]))
                #keep_prob = tf.placeholder(tf.float32)
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
                y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv

def train():
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
        #hostname=os.environ['HOSTNAME']
        #print("HOSTNAME::" + str(hostname))
        print("Job Name::" + str(FLAGS.job_name))
        print("Task Index::" + str(FLAGS.task_index))
        print("Batch Size::" + str(FLAGS.batch_size))
        is_chief = 0
        if FLAGS.job_name=='ps':
                print("In PS Task :: Creating Parameter Server")
                server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
                server.join()
                sys.exit(0)
        else:
                FLAGS.task_index=FLAGS.task_index-1
                server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
                with tf.device(tf.train.replica_device_setter(
        		worker_device="/job:worker/task:%d" % FLAGS.task_index,
	        	cluster=cluster)):
			run_metadata = tf.RunMetadata()
			x = tf.placeholder(tf.float32, [None, 784])
			y_ = tf.placeholder(tf.float32, [None, 10])
			x_image = tf.reshape(x, [-1,28,28,1])
			keep_prob = tf.placeholder(tf.float32)
			y_conv=model_forward(x_image,keep_prob)
			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
			global_step = tf.Variable(0)
			train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
			correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			saver = tf.train.Saver()
			init_op = tf.initialize_all_variables()
			summary_op = tf.summary.merge_all()
		sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/pylon5/cc5fp8p/asakhare/DistTF/logs_",#%d" % FLAGS.task_index,
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=60
				)
		mnist = input_data.read_data_sets(".", one_hot=True)
		times=[]
		with sv.managed_session(server.target,config=tf.ConfigProto(allow_soft_placement = True,log_device_placement=True)) as sess:
			step=0
			sess.as_default()
			t1_1 = datetime.datetime.now()
			while not sv.should_stop() and step < 1000:
				step=step+1
				#print("Step:" + str(step))
				batch_x, batch_y = mnist.train.next_batch(FLAGS.batch_size)
				start_time = time.time()
				_, loss = sess.run([train_step, global_step], {x: batch_x, y_: batch_y, keep_prob: 0.5})
				train_time_step = time.time() - start_time
				times.append(train_time_step)
				#if (step % 100) == 0:
                                train_accuracy = accuracy.eval(session=sess,feed_dict={x:batch_x, y_: batch_y, keep_prob: 1.0})
                                print("step %d, training accuracy %g"%(step, train_accuracy))
			t2_1 = datetime.datetime.now()
			print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
			print("Total Computation time: " + str(t2_1-t1_1))
		sv.stop()
		times = np.array(times)
		speeds = FLAGS.batch_size/times
		speed_mean = FLAGS.batch_size/np.mean(times)
		speed_uncertainty = np.std(speeds)/np.sqrt(float(len(speeds)))
		speed_jitter = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))
		print("Mean Time Per Step : " + str(np.mean(times)))
		print("Mean Speed : " + str(speed_mean) + " Images/Sec")
		print("speed uncertainty : " + str(speed_uncertainty))
		print("Speed Jitter : " + str(speed_jitter))

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.register("type", "bool", lambda v: v.lower() == "true")
        parser.add_argument(
      "--batch_size",
      type=int,
      default=64,
      help="Batch Size"
        )
        parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
        )
        parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Task Index Among Workers and PS"
        )
        FLAGS, unparsed = parser.parse_known_args()
        train()


