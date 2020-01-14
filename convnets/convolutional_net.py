import numpy as np
from sklearn.datasets import load_sample_images
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = np.array(load_sample_images().images, dtype=np.float32)

batch_size, height, width, channels = dataset.shape

filters_test = np.zeros(shape=(7,7,channels,2), dtype=np.float32)

filters_test[:,3,:,0] = 1
filters_test[:,3,:,1] = 1

#filters_test[:,:,:,1] = 1
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters_test, strides=[1,2,2,1], padding="SAME")
pooling = tf.nn.pool(convolution, [1,1], pooling_type="MAX", padding="SAME")

with tf.Session() as sess:
	output = sess.run(convolution, feed_dict={X:dataset})
	output1 = sess.run(pooling, feed_dict={convolution:output})

plt.imshow(output1[0, :, :,1])
plt.show()
