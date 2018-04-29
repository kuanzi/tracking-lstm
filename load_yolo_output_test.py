#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Power by Fu Hao
# 2017-10-24 12:25:47

import numpy as np
import os
import sys
def load_yolo_output_test (fold):
        paths = [os.path.join(fold,fn) for fn in next(os.walk(fold))[2]]
        paths = sorted(paths)
        print ("paths ",paths)
        # st= id
        # ed= id + batch_size*num_steps
        st= 1
        ed= 6
        paths_batch = paths[st:ed]
        print ("paths_batch ",paths_batch)
        yolo_output_batch= []
        ct= 0
        for path in paths_batch:
                ct += 1
                yolo_output = np.load(path)
                dirname = os.path.split(path)[0]
                name_no_ext= path.split("/")[-1]
                out_file = os.path.join(dirname, 'yolo_4', name_no_ext)
                print out_file
                yolo_output= np.reshape(yolo_output, 4102)
                yolo_output = yolo_output[4097:4101]
                np.save(out_file, yolo_output)
                # yolo_output_batch.append(yolo_output)
                # print ("batch: ", yolo_output_batch)
        # yolo_output_batch= np.reshape(yolo_output_batch, [batch_size*num_steps, 4102])
        # return yolo_output_batch

if __name__ == "__main__":
    print ('This is main of module "hello.py"')
    fold = sys.argv[1]
    load_yolo_output_test( fold )
