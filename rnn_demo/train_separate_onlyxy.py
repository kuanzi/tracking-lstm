 # -*- coding: utf-8 -*- 
from __future__ import print_function
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
from tensorflow.contrib import rnn
model_path='./tf_model/myonlyxy.ckpt'
#指定hog特征的维度
raw=1
col=2

# Parameters
learning_rate = 0.001
training_iters =2000000000
#batch_size = 18675
batch_size = 128
display_step =1

# Network Parameters
n_input = col 
n_steps = raw# timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 2 # total classes (0-9 digits)

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
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
    #return tf.matmul(outputs[-1], weights['out']) + biases['out'],outputs

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer()

import hog
import cv2
import numpy as np
import extension
data1,label1=extension.main_data_onlyxy()
# Launch the graph
print(data1.shape)
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
import copy
acclist = []
losslist = []

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        
        batch_x=data1       
        #batch_x=[[[0,0,0],[1,1,1],[0,0,0]]]
        batch_y=label1
	print('')   
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
	    tem_acc = copy.deepcopy(acc)
	    tem_loss = copy.deepcopy(loss)
	    acclist.append(tem_acc)
	    losslist.append(tem_loss)
	    #print(output,output.shape)
        step += 1
        save_path = saver.save(sess, model_path)
    print("Optimization Finished!")

txtName = 'train.txt'
f = file(txtName,'a+')
f.write('acc:'+str(acclist))
f.write('loss:'+str(losslist))


'''
img_dir='123.jpg'
imagg=cv2.imread(img_dir,cv2.IMREAD_GRAYSCALE)
a=hog.HOG_feature(imagg)
b=np.array(a)
print (b)
res = cv2.resize(b,(raw,col), interpolation = cv2.INTER_CUBIC)
bb=np.dstack(res)


batch_x2=data2
batch_y2=label2

saver = tf.train.Saver()

with tf.Session() as sess:
    load_path = saver.restore(sess, model_path)
    acc2=sess.run(accuracy, feed_dict={x:batch_x2, y:batch_y2})
    print('test-accuracy -- '+str(acc2))
'''
