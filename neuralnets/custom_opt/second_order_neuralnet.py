
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
####
# 1. Define the problem
####

learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# Here are 3 linear data points we'll want to fit on:
data_x = [0., 1., 2.]
data_y = [-1., 1., 3.]
batch_size = len(data_x)

# Input and Output. No batch_size for simplicity.
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Weight and bias.
# Computing hessians is currently only supported for one-dimensional tensors,
# so I did not bothered reshaping some 2D arrays for the parameters.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Making a prediction and comparing it to the true output
pred = tf.nn.softmax(tf.matmul(x,W) +b)
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# Preprocessings to the weight update
wrt_variables = [W, b]
grads = tf.gradients(loss, wrt_variables)

# The way I proceed here is equivalent to only compute the information
# contained in the diagonal of a single big hessian, because we isolated
# parameters from each others in the "wrt_variables" list.
hess = tf.hessians(loss, wrt_variables)
inv_hess = [tf.matrix_inverse(h) for h in hess]

# 2nd order weights update rule. Learning rate is of 1, because I
# trust the second derivatives obtained for such a small problem.
update_directions = [
    - tf.reduce_sum(h) * g
    for h, g in zip(inv_hess, grads)
]
op_apply_updates = [
    v.assign_add(up)
    for v, up in zip(wrt_variables, update_directions)
]


####
# 2. Proceed to solve the regression
####

# Initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# First loss
initial_loss = sess.run(
    loss,
    feed_dict={
        x: data_x,
        y: data_y
    }
)
print("Initial loss:", initial_loss)

# Weight and bias update, "training" phase:
for iteration in range(25):
    new_loss, _ = sess.run(
        [loss, op_apply_updates],
        feed_dict={
            x: data_x,
            y: data_y
        }
    )
    print("Loss after iteration {}: {}".format(iteration, new_loss))

# Results:
print("Prediction:", sess.run(pred, feed_dict={x: data_x}))
print("Expected:", data_y)
