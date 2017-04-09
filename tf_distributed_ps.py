from __future__ import print_function

import tensorflow as tf
import sys
import time

# cluster specification
parameter_servers = ["10.0.1.2:2222"]
workers = ["10.0.1.3:2222","10.0.1.4:2222","10.0.1.5:2222","10.0.1.6:2222","10.0.1.7:2222"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "ps", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS
# start a server for a specific task
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

# config
batch_size = 100
learning_rate = 0.0005
training_epochs = 100
logs_path = "/tmp/mnist/1"

# load mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

        # count the number of updates
        global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0), trainable = False)

        # input images
        with tf.name_scope('input'):
          # None -> batch size can be any size, 784 -> flattened mnist image
          x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
          # target 10 output classes
          y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

        # model parameters will change during training so we use tf.Variable
        tf.set_random_seed(1)
        with tf.name_scope("weights"):
                W1 = tf.Variable(tf.random_normal([784, 100]))
                W2 = tf.Variable(tf.random_normal([100, 10]))

        # bias
        with tf.name_scope("biases"):
                b1 = tf.Variable(tf.zeros([100]))
                b2 = tf.Variable(tf.zeros([10]))

        # implement model
        with tf.name_scope("softmax"):
                # y is our prediction
                z2 = tf.add(tf.matmul(x,W1),b1)
                a2 = tf.nn.sigmoid(z2)
                z3 = tf.add(tf.matmul(a2,W2),b2)
                y  = tf.nn.softmax(z3)

        # specify cost function
        with tf.name_scope('cross_entropy'):
                # this is our cost
                cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))

        # specify optimizer
        with tf.name_scope('train'):
                # optimizer is an "operation" which we can execute in a session
                grad_op = tf.train.GradientDescentOptimizer(learning_rate)
                train_op = grad_op.minimize(cross_entropy, global_step=global_step)

        with tf.name_scope('Accuracy'):
            # accuracy
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create a summary for our cost and accuracy
        tf.summary.scalar("cost", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)

        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), global_step=global_step, init_op=init_op)

    begin_time = time.time()
    frequency = 100
    with sv.prepare_or_wait_for_session(server.target) as sess:
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # perform training cycles
        start_time = time.time()
        for epoch in range(training_epochs):

            # number of batches in one epoch
            batch_count = int(mnist.train.num_examples/batch_size)

            count = 0
            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                # perform the operations we defined earlier on batch
                _, cost, summary, step = sess.run([train_op, cross_entropy, summary_op, global_step], feed_dict={x: batch_x, y_: batch_y})
                writer.add_summary(summary, step)

                count += 1
                if count % frequency == 0 or i+1 == batch_count:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Step: %d," % (step+1),
                                            " Epoch: %2d," % (epoch+1),
                                            " Batch: %3d of %3d," % (i+1, batch_count),
                                            " Cost: %.4f," % cost,
                                            " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
                    count = 0


            print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            print("Total Time: %3.2fs" % float(time.time() - begin_time))
            print("Final Cost: %.4f" % cost)

    sv.stop()
    print("done")

