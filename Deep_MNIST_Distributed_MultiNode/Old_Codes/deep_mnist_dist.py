import tensorflow as tf
import time
import os
import re
import sys
from tensorflow.examples.tutorials.mnist import input_data
import datetime
import numpy as np


batch_size = sys.argv[1]
print("Input Batch Size = " + str(batch_size))

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

#os.environ["CUDA_VISIBLE_DEVICES""] = ''
def model_forward(x_image,keep_prob):
	with tf.variable_scope('conv1') as scope:
        	W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
	        b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]))
	        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
	        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	with tf.variable_scope('conv2') as scope:
	        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
	        b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]))
	        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
	        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	with tf.variable_scope('dense1') as scope:
	        W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
	        b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]))
	        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	with tf.variable_scope('dense2') as scope:
	        W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
        	b_fc2 = tf.Variable(tf.constant(0.1,shape=[10]))
	        #keep_prob = tf.placeholder(tf.float32)
        	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	return y_conv

def tower_loss(cross_entropy_mean,scope):
	tf.add_to_collection('losses', cross_entropy_mean)
	losses = tf.get_collection('losses', scope)
	total_loss = tf.add_n(losses, name='total_loss')
	return total_loss

def average_gradients(tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = []
                for g, _ in grad_and_vars:
                        # Add 0 dimension to the gradients to represent the tower.
                        expanded_g = tf.expand_dims(g, 0)

                        # Append on a 'tower' dimension which we will average over below.
                        grads.append(expanded_g)

                # Average over the 'tower' dimension.
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)
                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)
        return average_grads


num_workers=len(d['worker'])
print("Number of workers = " + str(num_workers))
num_gpu = check_available_gpus()
print("Number of GPUs = " + str(num_gpu))
batch_splits=num_workers*num_gpu
print("Number of batch splits = " + str(batch_splits))
x  = tf.placeholder(tf.float32, [None, 784], name='x')
x_img=tf.reshape(x, [-1, 28, 28, 1])
x_dict={}
x_dict = dict(zip(['x'+str((i)) for i in range(batch_splits)],tf.split(x_img,batch_splits)))

y_dict={}
y = tf.placeholder(tf.float32, [None, 10],  name='y')
y_dict = dict(zip(['y'+str((i)) for i in range(batch_splits)],tf.split(y,batch_splits)))

keep_prob = tf.placeholder(tf.float32)

global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
opt=tf.train.AdamOptimizer(1e-4)
grads=[]
correct_predictions = []

count=0
with tf.variable_scope(tf.get_variable_scope()):
	for i in range(num_workers):
		for j in range(num_gpu):
			with tf.device(('/job:worker/task:{0}/gpu:{1}').format(i,j)):
				with tf.name_scope(('scope_task_{0}_gpu_{1}').format(i,j)) as infer_scope:	
					yy=model_forward(x_dict[('x{0}').format(count)],keep_prob)
					tf.get_variable_scope().reuse_variables()
					loss = tower_loss(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yy, labels=y_dict[('y{0}').format(count)])),infer_scope)
					grads.append(opt.compute_gradients(loss,tf.trainable_variables()))
					correct_predictions.append(tf.equal(tf.argmax(yy, 1), tf.argmax(y_dict[('y{0}').format(count)],1)))
					count+=1

print("Type of Grads : ")
print(grads)
grad = average_gradients(grads)
apply_grad_op = opt.apply_gradients(grad)

variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
# Group all updates to into a single train op.
train_op = tf.group(apply_grad_op, variables_averages_op)

accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
	sess.as_default()
	sess.run(tf.global_variables_initializer(),options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
	t1_1 = datetime.datetime.now()
	for step in range(0,5000):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		start_time = time.time()
		#sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
		_, loss_val = sess.run([train_op, loss], {x: batch_x, y: batch_y, keep_prob: 0.5})
		train_time_step = time.time() - start_time
		times.append(train_time_step)
		assert not np.isnan(loss_val), 'Model diverged with loss = NaN'
		if (step % 100) == 0:
			train_accuracy = accuracy.eval(feed_dict={x:batch_x, y: batch_y, keep_prob: 1.0})
			print("step %d, training accuracy %g"%(step, train_accuracy))
			#print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
	t2_1 = datetime.datetime.now()
	print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
	print("Total Computation time: " + str(t2_1-t1_1))

times = np.array(times)
speeds = batch_size/times
speed_mean = batch_size/np.mean(times)
speed_uncertainty = np.std(speeds)/np.sqrt(float(len(speeds)))
speed_jitter = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))

print("Mean Time Per Step : " + str(np.mean(times)))
print("Mean Speed : " + str(speed_mean) + " Images/Sec")
print("speed uncertainty : " + str(speed_uncertainty))
print("Speed Jitter : " + str(speed_jitter))


#with tf.device("/job:worker/task:0/cpu:0")
print("Finish")
#WIP
