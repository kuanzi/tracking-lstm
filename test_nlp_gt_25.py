# Copyright (c) <2016> <GUANGHAN NING>. All Rights Reserved.
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 

'''
Script File: ROLO_network_test_all.py

Description:

	ROLO is short for Recurrent YOLO, aimed at simultaneous object detection and tracking
	Paper: http://arxiv.org/abs/1607.05781
	Author: Guanghan Ning
	Webpage: http://guanghan.info/
'''

# Imports
import ROLO_utils as utils
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
# from tensorflow.models.rnn import rnn, rnn_cell
import cv2

import numpy as np
import os.path
import time
import random


class ROLO_TF:
    disp_console = True
    # restore_weights = True#False
    restore_weights = True

    # YOLO parameters
    fromfile = None
    # filewrite_img = False
    # filewrite_txt = False
    disp_console = True
    yolo_weights_file = 'weights/YOLO_small.ckpt'
    alpha = 0.1
    threshold = 0.2
    iou_threshold = 0.5
    num_class = 20
    num_box = 2
    grid_size = 7
    classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    w_img, h_img = [352, 240]

    # ROLO Network Parameters
    # rolo_weights_file = '/home1/fuhao/tracking/ROLO/data/model_demo.ckpt'
    # rolo_weights_file = '/home1/fuhao/tracking/ROLO/data/model_step1_exp2.ckpt'
    # rolo_weights_file = '/u03/Guanghan/dev/
    lstm_depth = 3
    num_steps = 3  # number of frames as an input sequence
    # num_feat = 4096
    num_predict = 4 # final output of LSTM 6 loc parameters
    num_gt = 4
    num_input = num_predict # data input: 4096+6= 5002

    # ROLO Parameters
    batch_size = 2
    display_step = 1

    # tf Graph input
    x = tf.placeholder("float32", [batch_size, num_steps, num_input])
    istate = tf.placeholder("float32", [batch_size, 2*num_input]) #state & cell => 2x num_input
    y = tf.placeholder("float32", [batch_size, num_gt, ])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_input, num_predict]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_predict]))
    }


    def __init__(self,argvs = []):
        print("ROLO init")
        self.ROLO(argvs)


    def LSTM_single(self, name,  _X, _istate, _weights, _biases):
        # with tf.device('/gpu:1'):
        # print ("before transpose: ", _X.get_shape().as_list())
        # input shape: (batch_size, n_steps, n_input)
        _X = tf.transpose(_X, [1, 0, 2])  # permute num_steps and batch_size
        # print ("after transpose: ", _X.get_shape().as_list())
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [self.num_steps, self.batch_size, self.num_input]) # (num_steps*batch_size, num_input)
        # _X = tf.reshape(_X, [ self.num_steps, self.batch_size, self.num_input]) # (num_steps*batch_size, num_input)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        # print ("after reshape: ", _X.get_shape().as_list())
        #_X = tf.split(_X, self.num_steps, 0) # n_steps * (batch_size, num_input)
        
        # # input shape: (batch_size, n_steps, n_input)
        # print ("before transpose: ", _X.get_shape().as_list())
        # _X = tf.transpose(_X, [1, 0, 2])  # permute num_steps and batch_size
        # print ("after transpose: ", _X.get_shape().as_list())
        # # Reshape to prepare input to hidden activation
        # _X = tf.reshape(_X, [self.num_steps * self.batch_size, self.num_input]) # (num_steps*batch_size, num_input)
        # # Split data because rnn cell needs a list of inputs for the RNN inner loop
        # print ("after reshape: ", _X.get_shape().as_list())
        # print (self.num_steps)
        # print (tf.cast(self.num_steps, tf.int32))
        # _X = tf.split(0, tf.cast(self.num_steps, tf.float32), _X) # n_steps * (batch_size, num_input)
        # print ("after split: ", _X.get_shape().as_list())

        #cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_input, self.num_input)
        cell = tf.nn.rnn_cell.LSTMCell(self.num_input, self.num_input)
        #state = _istate
        # for step in range(self.num_steps):
        #    # outputs, state = tf.nn.rnn(cell, [_X[step]], state)
        #   # print (cell)
        #   # print([_X[step]])
        #   # print(state)
        #    #outputs, state = tf.nn.static_rnn(cell = cell, inputs = [_X[step]], initial_state = state)
        print _X.get_shape().as_list()
        # print ("*********")
        outputs, state = tf.nn.dynamic_rnn(cell = cell, inputs = _X, initial_state = cell.zero_state(self.batch_size, tf.float32), time_major=True)
        print ("state:", state)
        print("output: ", outputs)
        # outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=_X_in, initial_state=state)
        # print ("############################33")
        # print(tf.get_variable_scope().original_name_scope)
        # print ("############################33")
        tf.get_variable_scope().reuse_variables()
        return outputs


    # Experiment with dropout
    def dropout_features(self, feature, prob):
        num_drop = int(prob * 4096)
        drop_index = random.sample(xrange(4096), num_drop)
        for i in range(len(drop_index)):
            index = drop_index[i]
            feature[index] = 0
        return feature
    '''---------------------------------------------------------------------------------------'''
    # def build_networks(self):
    #     if self.disp_console : print "Building ROLO graph..."
    #     # Build rolo layers
    #     self.lstm_module = self.LSTM_single('lstm_test', self.x, self.istate, self.weights, self.biases)
    #     self.ious= tf.Variable(tf.zeros([self.batch_size]), name="ious")
    #     self.sess = tf.Session()
    #     self.sess.run(tf.global_variables_initializer())
    #     self.saver = tf.train.Saver(tf.global_variables())
    #     ###
    #     self.saver.restore(self.sess, self.rolo_weights_file)
    #     if self.disp_console : print "Loading complete!" + '\n'


    def testing(self, x_path, y_path):
        total_loss = 0
        print("\n\n\ntesing_func\n\n")
        # Use rolo_input for LSTM training
        pred = self.LSTM_single('lstm_train', self.x, self.istate, self.weights, self.biases)
        print("pred: ", pred)
        self.feature1 = pred[0,0,:]
        
        # self.pred_location = pred[0][:, 4097:4101]
        self.pred_location = pred[0]
        print("pred_location: ", self.pred_location)
        print("self.y: ", self.y)
        self.correct_prediction = tf.square(self.pred_location - self.y)
        #print("self.correct_prediction: ", self.correct_prediction)
        self.accuracy = tf.reduce_mean(self.correct_prediction) * 100
        #print("self.accuracy: ", self.accuracy)
        #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.accuracy) # Adam Optimizer

        # Initializing the variables
        # init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        print ("\nLaunch the graph")
        saver = tf.train.Saver()
        with tf.Session() as sess:

            if (self.restore_weights == True):
                sess.run(init)
                # u1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1") 
                # u2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2") 
                # vars_to_rename = {
                # "lstm/basic_lstm_cell/weights": "lstm/basic_lstm_cell/kernel",
                # "lstm/basic_lstm_cell/biases": "lstm/basic_lstm_cell/bias",
                # }
                # new_checkpoint_vars = {}
                # reader = tf.train.NewCheckpointReader(FLAGS.checkpoint_path)
                # for old_name in reader.get_variable_to_shape_map():
                # if old_name in vars_to_rename:
                #     new_name = vars_to_rename[old_name]
                # else:
                #     new_name = old_name
                #     new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

                # init = tf.global_variables_initializer()
                saver.restore(sess, '/home1/fuhao/tracking/ROLO/checkpoints-only-25/pred-ep4999.ckpt')
                # saver = tf.train.Saver({"RNN/LSTMCell/W_0":'rnn/basic_lstm_cell/kernel', "RNN/LSTMCell/B": 'u2'})
                # self.saver.restore(sess, self.rolo_weights_file)
                print "Loading complete!" + '\n'
            else:
                sess.run(init)
            
            print ("directly, no load..........")
            id = 0 #don't change this
            total_time = 0.0
            #id= 1

            # Keep training until reach max iterations
            while id < self.testing_iters - self.num_steps:
                # Load training data & ground truth
                print ("# Load training data & ground truth")
                print ('x_path:', x_path)
                batch_xs = self.rolo_utils.load_yolo_output_test_4(x_path, self.batch_size, self.num_steps, id) # [num_of_examples, num_input] (depth == 1)

                print ("# Apply dropout to batch_xs")
                #for item in range(len(batch_xs)):
                #    batch_xs[item] = self.dropout_features(batch_xs[item], 0.4)

                batch_ys = self.rolo_utils.load_rolo_gt_test(y_path, self.batch_size, self.num_steps, id)
                batch_ys = utils.locations_from_0_to_1(self.w_img, self.h_img, batch_ys)

                # Reshape data to get 3 seq of 5002 elements
                batch_xs = np.reshape(batch_xs, [self.batch_size, self.num_steps, self.num_input])
                batch_ys = np.reshape(batch_ys, [self.batch_size, 4])
                #print("Batch_ys: ", batch_ys)

                start_time = time.time()
                feature1, pred_location, istate = sess.run([self.feature1, self.pred_location, self.istate],feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros((self.batch_size, 2*self.num_input))})
                print('feature:', feature1)
                print('pred_location:', pred_location)
                print('istate:', istate) 
                cycle_time = time.time() - start_time
                total_time += cycle_time

                print("ROLO Pred: ", pred_location)
                print("len(pred) = ", len(pred_location))
                print("ROLO Pred in pixel: ", pred_location[0][0]*self.w_img, pred_location[0][1]*self.h_img, pred_location[0][2]*self.w_img, pred_location[0][3]*self.h_img)
                print("correct_prediction int: ", (pred_location + 0.1).astype(int))

                print("# Save pred_location to file")
                print("output_path: ",self.output_path)
                print("pred_location: ", pred_location)
                print("batch_size: ", self.batch_size)
                utils.save_rolo_output_test(self.output_path, pred_location, id, self.num_steps, self.batch_size)

                #sess.run(optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros((self.batch_size, 2*self.num_input))})

                if id % self.display_step == 0:
                    # Calculate batch loss
                    loss = sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros((self.batch_size, 2*self.num_input))})
                    #print "Iter " + str(id*self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) #+ "{:.5f}".format(self.accuracy)
                    total_loss += loss
                id += 1
                #print(id)

            print "Testing Finished!"
            avg_loss = total_loss/id
            print "Avg loss: " + str(avg_loss)
            print "Time Spent on Tracking: " + str(total_time)
            print "fps: " + str(id/total_time)
            #save_path = self.saver.save(sess, self.rolo_weights_file)
            #print("Model saved in file: %s" % save_path)

        return None


    def ROLO(self, argvs):

            self.rolo_utils= utils.ROLO_utils()
            self.rolo_utils.loadCfg()
            self.params = self.rolo_utils.params

            arguments = self.rolo_utils.argv_parser(argvs)

            if self.rolo_utils.flag_train is True:
                self.training(utils.x_path, utils.y_path)
            elif self.rolo_utils.flag_track is True:
                self.build_networks()
                self.track_from_file(utils.file_in_path)
            elif self.rolo_utils.flag_detect is True:
                self.build_networks()
                self.detect_from_file(utils.file_in_path)
            else:
                print "Default: running ROLO test."
                # self.build_networks()

                evaluate_st = 26
                evaluate_ed = 29

                for test in range(evaluate_st, evaluate_ed + 1):

                    [self.w_img, self.h_img, sequence_name, dummy_1, self.testing_iters] = utils.choose_video_sequence(test)

                    x_path = os.path.join('benchmark/DATA', sequence_name, 'yolo_out/', 'yolo_4/')
                    y_path = os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
                    self.output_path = os.path.join('benchmark/DATA', sequence_name, 'rolo_out_test-rnn-only-25/')
                    utils.createFolder(self.output_path)

                    #self.rolo_weights_file = '/u03/Guanghan/dev/ROLO-dev/output/ROLO_model/model_nodrop_30_2.ckpt'  #no dropout
                    #self.rolo_weights_file = '/u03/Guanghan/dev/ROLO-dev/output/ROLO_model/model_online.ckpt'
                    #self.rolo_weights_file = '/u03/Guanghan/dev/ROLO-dev/output/MOLO/model_MOT.ckpt'
                    #self.rolo_weights_file = '/u03/Guanghan/dev/ROLO-dev/output/MOLO/model_MOT_0.2.ckpt'

                    #self.rolo_weights_file= '/u03/Guanghan/dev/ROLO-dev/output/ROLO_model/model_step6_exp0.ckpt'
                    #self.rolo_weights_file= '/u03/Guanghan/dev/ROLO-dev/output/ROLO_model/model_step3_exp1.ckpt'
                    #self.rolo_weights_file= '/u03/Guanghan/dev/ROLO-dev/output/ROLO_model/model_step6_exp2.ckpt'

                    #self.rolo_weights_file= '/u03/Guanghan/dev/ROLO-dev/output/ROLO_model/model_step3_exp2.ckpt'
                    #self.rolo_weights_file= '/u03/Guanghan/dev/ROLO-dev/output/ROLO_model/model_step9_exp2.ckpt'
                    #self.rolo_weights_file= '/u03/Guanghan/dev/ROLO-dev/output/ROLO_model/model_step1_exp2.ckpt'

                    # self.rolo_weights_file= '/home1/fuhao/tracking/ROLO/data/model_demo.ckpt'

                    self.num_steps = 3  # number of frames as an input sequence
                    print("TESTING ROLO on video sequence: ", sequence_name)
                    self.testing(x_path, y_path)


    '''----------------------------------------main-----------------------------------------------------'''
def main(argvs):
        ROLO_TF(argvs)


if __name__=='__main__':
        main(' ')

