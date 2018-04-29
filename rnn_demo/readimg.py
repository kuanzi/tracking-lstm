# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 09:25:11 2017

@author: Administrator
"""

import matplotlib.image as mpimg
import os
import cv2
def readseq(path):
    seq = []
    for filename in os.listdir(path):   #listdir的参数是文件夹的路径
        #print ( filename)                                  #此时的filename是文件夹中文件的名称
        path_tem = path + '/' + filename      #每一图片路径
        #tem_frame = mpimg.imread(path_tem)#读取每一图片
        tem_frame = cv2.imread(path_tem, cv2.IMREAD_GRAYSCALE)
    #plt.imshow(tem_frame)
        seq.append(tem_frame)

    return (seq)


def croppatch( img, pos, boxw, boxh):
        pos0, pos1 = int(pos[0]), int(pos[1])
        tem_patch = img[pos0 - boxh/2:pos0 + boxh/2, pos1 - boxw/2:pos1 + boxw/2]
        return tem_patch
    
