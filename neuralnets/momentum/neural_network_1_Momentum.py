import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("/tmp/data/")
n_inputs = 28*28
n_hidden1 = 3000
n_hidden2 = 1000
n_hidden3 = 1000
n_outputs = 10
n_epochs = 400
batch_size = 10
momentum = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

def neuron_layer(X, n_neurons, name, activation=None):
	with tf.name_scope(name):
		n_inputs = int(X.get_shape()[1])
		stddev = 2/np.sqrt(n_inputs)
		init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
		W = tf.Variable(init, name="Weights")
		b = tf.Variable(tf.zeros([n_neurons]), name="biases")
		z = tf.matmul(X,W)+b
		if activation =="relu":
	    		return tf.nn.relu(z)
		else:
	    		return z

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    hidden3 = neuron_layer(hidden2, n_hidden3, "hidden3", activation="sigmoid")
    logits = neuron_layer(hidden3, n_outputs, "outputs")

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X,n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1,n_hidden2, scope="hidden2")
    hidden3 = fully_connected(hidden2, n_hidden3, scope="hidden3")
    logits = fully_connected(hidden3, n_outputs, scope="outputs", activation_fn=None)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct= tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		for iteration in range(mnist.train.num_examples//batch_size):
			X_batch, y_batch = mnist.train.next_batch(batch_size)
			sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})
		acc_train = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})
		acc_test = accuracy.eval(feed_dict ={X:mnist.test.images, y:mnist.test.labels})
		print(epoch, "Train accuracy:", acc_train,"Test accuracy:", acc_test)
	save_path = saver.save(sess, "./my_model_final.ckpt")
