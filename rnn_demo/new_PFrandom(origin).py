# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 09:48:41 2017

@author: Administrator
"""

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


sizeim = [int(frame0.shape[0]),int(frame0.shape[1])]#图片size

print(sizeim)
leftcenter0 = array([boxh/2,boxw/2])
rightcenter1 = array([sizeim[0]-boxh/2,sizeim[1]-boxw/2]) 

#扫描选取patch

feature = []

possave=[]

radius = 16#dense半径


pos0 = array([136, 223])#第一帧中心


def indata(radius,pos,sizeim,boxh,boxw,frame):
    
    fpos=[]

    import hog
    feature = []
    postem=array([136, 223])
    
    for i in range(radius):
        postem[0] = pos[0] + i
        if postem[0]<sizeim[0]-boxh/2:#检查边界
    
            for j in range(radius):
                postem[1] = pos[1] + j
                              
                if postem[1]<sizeim[1]-boxh/2:#检查边界
    
		    pos1=array([pos[0]+i,pos[1]+j])
		    fpos.append(pos1)

                    patch = croppatch(frame,postem,boxw,boxh)#patch抓取
                    feature_particle = hog.HOG_feature(patch)#特征提取
                    feature_2=np.array(feature_particle)
                    feature.append(feature_2)
                    
    for i in range(radius):
        postem[0] = pos[0] - i
        if postem[0]>boxh/2:#检查边界
    
            for j in range(radius):
                postem[1] = pos[1] + j
                              
                if postem[1]<sizeim[1]-boxh/2:#检查边界
		    
		    pos1=array([pos[0]-i,pos[1]+j])
		    fpos.append(pos1)
    
                    patch = croppatch(frame,postem,boxw,boxh)#patch抓取
                    feature_particle = hog.HOG_feature(patch)#特征提取
                    feature_2=np.array(feature_particle)
                    feature.append(feature_2)

    for i in range(radius):
        postem[0] = pos[0] - i
        if postem[0]>boxh/2:#检查边界
    
            for j in range(radius):
                postem[1] = pos[1] - j
                              
                if postem[1]>boxh/2:#检查边界

		    pos1=array([pos[0]-i,pos[1]-j])
		    fpos.append(pos1)
    
                    patch = croppatch(frame,postem,boxw,boxh)#patch抓取
                    feature_particle = hog.HOG_feature(patch)#特征提取
                    feature_2=np.array(feature_particle)
                    feature.append(feature_2)
                    
    for i in range(radius):
        postem[0] = pos[0] + i
        if postem[0]<sizeim[0]-boxh/2:#检查边界
    
            for j in range(radius):
                postem[1] = pos[1] - j
                              
                if postem[1]>boxh/2:#检查边界

		    pos1=array([pos[0]+i,pos[1]-j])
		    fpos.append(pos1)
    
                    patch = croppatch(frame,postem,boxw,boxh)#patch抓取
                    feature_particle = hog.HOG_feature(patch)#特征提取
                    feature_2=np.array(feature_particle)
                    feature.append(feature_2) 
    return feature,fpos

[featuretest,postest] = indata(radius,pos0,sizeim,boxh,boxw,frame0)

feature = np.array(featuretest)

tem_w=cnc_rnn.predict(feature)

count = 0
iniw = zeros(tem_w.shape[0])
for each_one in tem_w:
    
    w = 1-((1-each_one[0])**2+(0- each_one[1])**2)**(0.5)
    iniw[count]=iniw[count]+w
    count += 1

iniw /= sum(iniw)
from itertools import izip
posx=0
posy=0
for testw,testpos in izip(iniw,postest):
    #print testw,testpos
    posx = posx + testpos[0]*testw
    posy = posy + testpos[1]*testw
posx = int(posx)
posy = int(posy)


print posx,posy
#print(postest)

