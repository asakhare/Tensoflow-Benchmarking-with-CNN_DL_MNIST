from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import datetime
import time
import numpy as np
import sys

batch_size = int(sys.argv[1])
print("Input Batch Size=" + str(batch_size))
print("Execution Type: Asynchronous")

#Make Sure it sees the devices
from tensorflow.python.client import device_lib
print("*******Device Info Seen By TensorFlow*******")
print(device_lib.list_local_devices())
print("********************************************")
#times = []

def check_available_gpus():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
    gpu_num = len(gpu_names)

    print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))

    return gpu_num

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

def conv2d(xx, W):
    return tf.nn.conv2d(xx, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(xx):
    return tf.nn.max_pool(xx, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


def model_forward(xx,keep_prob):
	w0=tf.get_variable('w0',initializer=tf.truncated_normal([5, 5,1,32], stddev=0.1),trainable=True)
	b0=tf.get_variable('b0',initializer=tf.zeros([32]),trainable=True)

	w1=tf.get_variable('w1',initializer=tf.truncated_normal([5,5,32,64], stddev=0.1),trainable=True)
	b1=tf.get_variable('b1',initializer=tf.zeros([64]),trainable=True)

	w2=tf.get_variable('w2',initializer=tf.truncated_normal([7*7*64,1024], stddev=0.1),trainable=True)
	b2=tf.get_variable('b2',initializer=tf.zeros([1024]),trainable=True)

	w3=tf.get_variable('w3',initializer=tf.truncated_normal([1024,10], stddev=0.1),trainable=True)
	b3=tf.get_variable('b3',initializer=tf.zeros([10]),trainable=True)

    	h_conv1=tf.nn.relu(conv2d(xx,w0)+b0);
	h_pool1=max_pool_2x2(h_conv1)

	h_conv2=tf.nn.relu(conv2d(h_pool1,w1)+b1);
	h_pool2=max_pool_2x2(h_conv2)

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w2)+b2)

	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	y = tf.matmul(h_fc1_drop,w3)+b3
	return y

def tower_loss(cross_entropy_mean,scope):
	tf.add_to_collection('losses', cross_entropy_mean)
	losses = tf.get_collection('losses', scope)
	total_loss = tf.add_n(losses, name='total_loss')
	return total_loss

def run_with_location_trace(sess, op):
	# From https://stackoverflow.com/a/41525764/7832197
	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	sess.run(op, options=run_options, run_metadata=run_metadata)
	for device in run_metadata.step_stats.dev_stats:
		print(device.device)
		for node in device.node_stats:
			print("  ", node.node_name)
	
def train():
	with tf.Graph().as_default(),tf.device('/cpu:0'):
		gpu_num = check_available_gpus()
		#batch_size = 1024
		mnist = input_data.read_data_sets(".", one_hot=True)
		times = []
		run_metadata = tf.RunMetadata()

		x  = tf.placeholder(tf.float32, [None, 784], name='x')
		x_img=tf.reshape(x, [-1, 28, 28, 1])
		x_dict={}
		x_dict = dict(zip(['x'+str((i)) for i in range(gpu_num)],tf.split(x_img,gpu_num)))
		
		y_dict={}
		y = tf.placeholder(tf.float32, [None, 10],  name='y')
		y_dict = dict(zip(['y'+str((i)) for i in range(gpu_num)],tf.split(y,gpu_num)))
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		opt=tf.train.AdamOptimizer(1e-4)
		keep_prob = tf.placeholder(tf.float32)
		grads=[]
		correct_predictions = []
		with tf.variable_scope(tf.get_variable_scope()):	
			for i in range(0,gpu_num):
					with tf.device(('/gpu:{0}').format(i)):
						with tf.name_scope(('scope_gpu_{0}').format(i)) as infer_scope:
							#batch_x, batch_y = mnist.train.next_batch(batch_size)
							yy=model_forward(x_dict[('x{0}').format(i)],keep_prob)
							tf.get_variable_scope().reuse_variables()
							loss = tower_loss(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yy, labels=y_dict[('y{0}').format(i)])),infer_scope)
							grads.append(opt.compute_gradients(loss,tf.trainable_variables()))
							correct_predictions.append(tf.equal(tf.argmax(yy, 1), tf.argmax(y_dict[('y{0}').format(i)],1)))

		grad = average_gradients(grads)
		apply_grad_op = opt.apply_gradients(grad)

		variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
		variables_averages_op = variable_averages.apply(tf.trainable_variables())
		# Group all updates to into a single train op.
		train_op = tf.group(apply_grad_op, variables_averages_op)
		
		#correct_prediction = tf.equal(tf.argmax(yy, 1), tf.argmax(y_dict['y0'], 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
    		with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
			sess.as_default()
			sess.run(tf.global_variables_initializer(),options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
			t1_1 = datetime.datetime.now()
			for step in range(0,5000):
				batch_x, batch_y = mnist.train.next_batch(batch_size*gpu_num)
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
			#from tensorflow.python.client import timeline
			#trace = timeline.Timeline(step_stats=run_metadata.step_stats)
			#trace_file = open('timeline.ctf.json', 'w')
			#trace_file.write(trace.generate_chrome_trace_format())
			#run_with_location_trace(sess,train_op)
			#print(sess.run(b0))
			print(tf.GraphKeys.TRAINABLE_VARIABLES)
			writer = tf.summary.FileWriter('./my_graph',sess.graph)
			writer.close()
			for device in run_metadata.step_stats.dev_stats:
				print(device.device)
				for node in device.node_stats:
					print("  ", node.node_name)

	times = np.array(times)
	speeds = batch_size/times
	speed_mean = batch_size/np.mean(times)
	speed_uncertainty = np.std(speeds)/np.sqrt(float(len(speeds)))
	speed_jitter = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))

	print("Mean Time Per Step : " + str(np.mean(times)))
	print("Mean Speed : " + str(speed_mean) + " Images/Sec")
	print("speed uncertainty : " + str(speed_uncertainty))
	print("Speed Jitter : " + str(speed_jitter))
		
def main():
    train()

if __name__ == "__main__":
    main()
