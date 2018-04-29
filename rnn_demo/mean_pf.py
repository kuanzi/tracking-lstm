# -*- coding: utf-8 -*-
"""
Created on Fri Jun 02 18:12:48 2017

@author: Administrator
"""

from skimage import io,data
from numpy import *  
from numpy.random import *  
###################HOG提取函数
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import math
from itertools import izip
import hog
import cnc_rnn

patchresult = []
#切patch
def croppatch( img, pos, boxw, boxh):
        tem_patch = img[pos[0] - boxh/2:pos[0] + boxh/2, pos[1] - boxw/2:pos[1] + boxw/2]
        return tem_patch
    
    
boxh = 46
boxw = 46

frame0 = cv2.imread('../ball/img/img0002.jpg', cv2.IMREAD_GRAYSCALE)
#frame0 = cv2.imread('H:/RPF/ball/img/img0002.jpg', cv2.IMREAD_GRAYSCALE)#读入第一帧图片
#frame0 = np.sqrt(frame0 / float(np.max(frame0)))#gamma滤波

#plt.imshow(frame0)

[winy,winx]=[int(frame0.shape[0]/boxh*2),int(frame0.shape[1]/boxw*2)]#窗口数
sizeim = [int(frame0.shape[0]),int(frame0.shape[1])]


leftcenter0 = array([boxh/2,boxw/2])
rightcenter1 = array([sizeim[0]-boxh/2,sizeim[1]-boxw/2]) 

#两次扫描选取patch

feature = []

possave=[]
for i in range(winy-1):
    
    #print i
    pos = leftcenter0 + [boxh/2*i,0]
    #print pos
    
    for j in range(winx-1):
        pos0 =pos+[0,boxw/2*j]
        possave.append(pos0)
        #print(pos0)
        patch = croppatch(frame0,pos0,boxw,boxh)
        feature_particle = hog.HOG_feature(patch)
        feature_2=np.array(feature_particle)
        #print(feature_2.shape)
        feature.append(feature_2)
        #tem_w=cnc_rnn.predict(feature_2)
        #w = 1-((1-tem_w[0][0])**2+(0- tem_w[0][1])**2)**(0.5)
        #print w
	#if w>0.5:
	    #patchresult.append(pos0)
        #plt.imshow(patch)
        #plt.show()
        
for i in range(winy-1):
    
    #print i
    pos = rightcenter1 - [boxh/2*i,0]
    #print pos
    
    for j in range(winx-1):
        pos0 =pos-[0,boxw/2*j]
        possave.append(pos0)
        #print(pos0)
        patch = croppatch(frame0,pos0,boxw,boxh)
        feature_particle = hog.HOG_feature(patch)
        feature_2=np.array(feature_particle)
        #print(feature_2.shape)
        feature.append(feature_2)
        #tem_w=cnc_rnn.predict(feature_2)
        #w = 1-((1-tem_w[0][0])**2+(0- tem_w[0][1])**2)**(0.5)
        #print w
	#if w>0.5:
	    #patchresult.append(pos0)
        #print patch.shape
        #plt.imshow(patch)
        #plt.show()

feature = np.array(feature)

tem_w=cnc_rnn.predict(feature)

count = 0
for each_one in tem_w:
    
    w = 1-((1-each_one[0])**2+(0- each_one[1])**2)**(0.5)
    if w>0.5:
        print w
	print count
	print possave[count]

    count+=1
#print patchresult

