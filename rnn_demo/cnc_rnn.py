 # -*- coding: utf-8 -*- 
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import hog
import cv2
import numpy as np
import sample
import datetime

model_path='./tf_model/my.ckpt'
#指定hog特征的维度
raw=1
col=512

# Network Parameters
n_input = col 
n_steps = raw# timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 2 # total classes 0/1

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])

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

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
test=tf.nn.softmax(pred)
saver = tf.train.Saver()


def predict(feature):
    with tf.Session() as sess:
        load_path = saver.restore(sess, model_path)
	begin = datetime.datetime.now()
        prediction=sess.run(test,feed_dict={x:feature})
	end = datetime.datetime.now()
	print(end-begin)
        print (prediction)
        return (prediction)
