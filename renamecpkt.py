#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Power by Fu Hao
# 2017-10-16 18:23:58
import tensorflow as tf

def RenameCkpt():
# 1.0.1 : 1.2.1
    vars_to_rename = {
    "lstm/basic_lstm_cell/weights": "lstm/basic_lstm_cell/kernel",
    "lstm/basic_lstm_cell/biases": "lstm/basic_lstm_cell/bias",
    }
    new_checkpoint_vars = {}
    reader = tf.train.NewCheckpointReader(tf.FLAGS.checkpoint_path)
    for old_name in reader.get_variable_to_shape_map():
      if old_name in vars_to_rename:
        new_name = vars_to_rename[old_name]
      else:
        new_name = old_name
      new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(new_checkpoint_vars)

    with tf.Session() as sess:
      sess.run(init)
      saver.save(sess, "/home/ndscbigdata/work/change/tf/gan/im2txt/ckpt/newmodel.ckpt-2000000")
    print("checkpoint file rename successful... ")

if __name__ == '__main__':
    RenameCkpt()
