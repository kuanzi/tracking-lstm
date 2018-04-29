# -*- coding: utf-8 -*-
"""
Created on Fri Aug 04 17:29:42 2017

@author: Administrator
"""

def save_out(filepath, out_seq):
    from skimage import io
    
    length = len(out_seq)
    for i in range(length):
        
        if i < 10:
            filename = 'test000' + str(i) + '.jpg'
        elif 10 <= i <100:
            filename = 'test00' + str(i) + '.jpg'
        elif 100 <= i <1000:
            filename = 'test0' + str(i) + '.jpg'
        else:
            print('out of range!')
            break
        path = filepath + filename
        io.imsave(path, out_seq[i])