# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 09:22:28 2017

@author: runqing
"""

from __future__ import print_function
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
from tensorflow.contrib import rnn
model_path='./tf_model/my.ckpt'
import numpy
raw=1
col=4

# Parameters
learning_rate = 0.001
training_iters =2000000000
#batch_size = 18675
batch_size = 128
display_step =1

# Network Parameters
n_input = col 
n_steps = raw# timesteps
n_hidden = 4 # hidden layer num of features
n_classes = 4 # total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    print(outputs)
    # Linear activation, using rnn inner loop last output
    return outputs
    #return tf.matmul(outputs[-1], weights['out']) + biases['out']
    #return tf.matmul(outputs[-1], weights['out']) + biases['out'],outputs

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()



#import train_separate
import extension
data1,label1=extension.test_data()

batch_x2=data1
batch_y2=label1

#saver = tf.train.Saver()

model_path = './tf_model/my.ckpt'
test=tf.nn.softmax(pred)

with tf.Session() as sess:
    load_path = saver.restore(sess, model_path)
    #acc2=sess.run(accuracy, feed_dict={x:batch_x2, y:batch_y2})
    prediction=sess.run(pred, feed_dict={x:batch_x2})
    #print('test-accuracy -- '+str(acc2))
    print(batch_x2.shape,batch_y2.shape)
print(prediction)
result = prediction[0]
print(result.shape)
txtName = 'test_separate.txt'
#f = file(txtName,'a+')
#f.writelines(prediction)
numpy.savetxt(txtName,result)
