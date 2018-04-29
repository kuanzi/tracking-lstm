# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

#GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

#data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print mnist.train.images.shape

#parameter

lr = 1e-3
#batch_size(placeholder)
batch_size = tf.placeholder(tf.int32) #type:int32
#batch_size = 128
#each T input size:28(a row:there is 28 pixels in a row)
input_size = 28
#time of all the T is 28(each image has 28 rows)
timestep_size = 28
#nodes of each hidden layer
hidden_size = 256
#LSTM layer number
layer_num = 2
#number of classification(when regression:1)
class_num = 10

_X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32)


#model of LSTM

#step1:the input of RNN:traslate 784bit data into 28*28 size image
#the input of RNN:shape = (batch_size, timestep_size, input_size)
X = tf.reshape(_X, [-1,28,28])
#print X.shape
#step2:fist layer LSTM_cell:only set the hidden_size, and it can auto get X size
lstm_cell = rnn.BasicLSTMCell(num_units = hidden_size, forget_bias=1.0, state_is_tuple=True)
#step3:dropout layer:only set output_keep_prob
lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
#step4:realize muti-layer LSTM by MultiRNNCell
mlstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
#step5:initial state(zero)
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
#step6:run the two-hidden-layer LSTM
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
	if timestep > 0:
	    tf.get_variable_scope().reuse_variables()
	#state save each layer's state of LTSM
	(cell_output, state) = mlstm_cell(X[:, timestep, :], state)
	outputs.append(cell_output)
h_state=outputs[-1]
#the output of h_state come from hidden cell


#train and test
#to classify objects,we need a softmax layer
#set softmax layer's weight and bias
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)

#loss and evalution function
cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


sess.run(tf.global_variables_initializer())
for i in range(2000):
    _batch_size = 128
    batch = mnist.train.next_batch(_batch_size)
    if (i+1)%200 == 0:
	train_accuracy = sess.run(accuracy, feed_dict={ _X:batch[0], y:batch[1], keep_prob: 1.0, batch_size: _batch_size})
	#the epoch number that was completed
	print('Iter%d, step %d, training accuracy %g' % ( mnist.train.epochs_completed, (i+1), train_accuracy))
    sess.run(train_op, feed_dict={_X:batch[0], y: batch[1], keep_prob:0.5, batch_size: _batch_size})

#test accuracy
print 'test accuracy %g' % sess.run(accuracy, feed_dict={_X: mnist.test.images, y:mnist.test.labels, keep_prob: 1.0, batch_size:mnist.test.images.shape[0]})

print mnist.test.images.shape

import matplotlib.pyplot as plt
print mnist.train.labels[4]

X3 = mnist.train.images[4]
img3 = X3.reshape([28,28])
X3.shape = [-1, 784]
y_batch = mnist.train.labels[0]
print y_batch
y_batch.shape = [-1, class_num]

X3_outputs = np.array(sess.run(outputs, feed_dict={_X:X3,y:y_batch, keep_prob:1.0, batch_size: 1}))
print X3_outputs.shape

print mnist.test.images.shape
print mnist.test.labels.shape
