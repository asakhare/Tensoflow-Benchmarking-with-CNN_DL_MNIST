import argparse
import sys

import tensorflow as tf

FLAGS = None


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      #loss = ...
      #global_step = tf.contrib.framework.get_or_create_global_step()

      #train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
	  
	  ############################################Model#######################################
	from tensorflow.examples.tutorials.mnist import input_data
	times = []
	batch_size = 50
	mnist = input_data.read_data_sets(".", one_hot=True)

	x = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32, [None, 10])

	x_image = tf.reshape(x, [-1,28,28,1])

	W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
	b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]))
	h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
	h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
	b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]))
	h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
	h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
	b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]))
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
	b_fc2 = tf.Variable(tf.constant(0.1,shape=[10]))
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()

	sess.run(tf.global_variables_initializer())
	for i in range(20000):
		batch = mnist.train.next_batch(batch_size)
		if i%1000 == 0:
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
#####################################################################################################

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See <a href="../api_docs/python/tf/train/SyncReplicasOptimizer"><code>tf.train.SyncReplicasOptimizer</code></a> for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        mon_sess.run(train_op)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
