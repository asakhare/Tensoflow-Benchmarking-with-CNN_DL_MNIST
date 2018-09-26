#This example trains a convolutional neural network (CNN) on the MNIST dataset.
#It is primarily taken from the TensorFlow tutorial with small modifications.
#In particular, the deprecated tf.nn.softmax_cross_entropy_with_logits(...) is 
#replaced with tf.nn.softmax_cross_entropy_with_logits_v2(...). 

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time
import numpy as np

#Make Sure it sees the devices
from tensorflow.python.client import device_lib
print("*******Device Info Seen By TensorFlow*******")
print(device_lib.list_local_devices())
print("********************************************")

times = []
batch_size = 64
mnist = input_data.read_data_sets(".", one_hot=True)
run_metadata = tf.RunMetadata()

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

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
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#keep_prob = tf.placeholder(tf.float32)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

num_cores = int(os.environ["SLURM_NTASKS"])
config = tf.ConfigProto(log_device_placement=True)
config.intra_op_parallelism_threads = 1 #num_cores
config.inter_op_parallelism_threads = 1 #num_cores
print("Number of CPU Cores = " + str(num_cores))
sess = tf.InteractiveSession(config=config)

sess.run(tf.global_variables_initializer(),options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
for i in range(5000):
  batch = mnist.train.next_batch(batch_size)
  if i%10 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  start_time = time.time()
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  train_time_step = time.time() - start_time
  times.append(train_time_step)

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

times = np.array(times)
speeds = batch_size/times
speed_mean = batch_size/np.mean(times)
speed_uncertainty = np.std(speeds)/np.sqrt(float(len(speeds)))
speed_jitter = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))

print("Mean Time Per Step : " + str(np.mean(times)))
print("Mean Speed : " + str(speed_mean) + " Images/Sec")
print("speed uncertainty : " + str(speed_uncertainty))
print("Speed Jitter : " + str(speed_jitter))

from tensorflow.python.client import timeline
trace = timeline.Timeline(step_stats=run_metadata.step_stats)
trace_file = open('timeline.ctf.json', 'w')
trace_file.write(trace.generate_chrome_trace_format())

#Write Graph
writer = tf.summary.FileWriter('./my_graph',sess.graph)
writer.close()

print("Device Placement ::")
for device in run_metadata.step_stats.dev_stats:
	print(device.device)
	for node in device.node_stats:
		print("  ", node.node_name)      
