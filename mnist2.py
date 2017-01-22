from __future__ import print_function

import tensorflow as tf

tf.reset_default_graph()

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops,init_ops
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

random_normal= init_ops.random_normal_initializer(mean=0.0,stddev=1.0,dtype=tf.float32)


##NEW##   #use scopes and get_variable to create variables on the fly
def _conv2d_layer(x,scope,output_size,strides=1):
	input_size=x.get_shape().as_list()[-1]
	with vs.variable_scope(scope):
		W = vs.get_variable("Matrix", [5,5,input_size,output_size],initializer=random_normal)
		_x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
		b=vs.get_variable('Bias', [output_size],
			initializer=random_normal)
		_x=tf.nn.bias_add(_x,b)
		return tf.nn.relu(_x)

##NEW## added from my collection in folder TF/
from TF.utils import _simple_relu_layer,_linear

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model##NEW## replaces conv_net,weights, and biases section
def conv_net(x, dropout):
	# Reshape input picture
	x_ = tf.reshape(x, shape=[-1, 28, 28, 1])
	x_=_conv2d_layer(x_,'conv1',32,strides=1)
	x_ = maxpool2d(x_, k=2)
	x_=_conv2d_layer(x_,'conv2',64,strides=1)
	tf.summary.histogram('conv2_activation',x_)#NEW#summary
	x_ = maxpool2d(x_, k=2)
	x_=tf.reshape(x_,[-1,7*7*64])
	x_=_simple_relu_layer(x_,scope='fc1',output_size=1024)
	x_=tf.nn.dropout(x_, dropout)
	x_=_linear(x_,n_classes,'projection')
	return x_
	

# Construct model
pred = conv_net(x, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
##NEW## summary
tf.summary.scalar('loss',cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuarcy',accuracy)#NEW# summary
tf.summary.scalar('loss',cost)


merged = tf.summary.merge_all()

writer=tf.summary.FileWriter('models/')
writer=tf.train.SummaryWriter('models/')

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
		# Calculate batch loss and accuracy
		loss, acc,summary = sess.run([cost, accuracy,merged], feed_dict={x: batch_x,#NEW#
						      y: batch_y,
						      keep_prob: 1.})
		writer.add_summary(summary,step)##NEW##
		writer.flush()
		print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
		"{:.6f}".format(loss) + ", Training Accuracy= " + \
		"{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))
