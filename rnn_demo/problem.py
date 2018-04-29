# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:46:31 2017

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
import copy

frame0 = cv2.imread('../ball/img/img0594.jpg', cv2.IMREAD_GRAYSCALE)

def croppatch( img, pos, boxw, boxh):
        tem_patch = img[pos[0] - boxh/2:pos[0] + boxh/2, pos[1] - boxw/2:pos[1] + boxw/2]
        return tem_patch
    
    
boxh = 46
boxw = 46

#plt.imshow(frame0)

[i,j] = frame0.shape
i,j = int(i),int(j)

pos_target = array([136, 223])
pospixel = array([boxh/2, boxw/2])

feature = []
pos = []

zero=[]
zero_feature = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
for m in range(16):
    zero.append(zero_feature)


for pix in range(i-boxh+1):
    for piy in range(j-boxw+1):
        patch = croppatch(frame0,pospixel,boxw,boxh)
        
        try:
            
            feature_particle = hog.HOG_feature(patch)#特征提取
        except:
            feature_particle = zero
            
        feature_2=np.array(feature_particle)
        feature.append(feature_2)
	
	postem = copy.deepcopy(pospixel)
        pos.append(postem)
        
        pospixel[1]+=1
        #print pospixel
    pospixel[0]+=1
    pospixel[1]=boxw/2
            

feature = np.array(feature)

import cnc_rnn
#tem_w=cnc_rnn.predict(feature)
tem_w=cnc_rnn.predict(feature)

count = 0
totalproblem = 0
iniw = zeros(tem_w.shape[0])

txtName = 'test.txt'
f = file(txtName,'a+')

for each_one in tem_w:
        
    w = 1-((1-each_one[0])**2+(0- each_one[1])**2)**(0.5)
    iniw[count]=iniw[count]+w
    if w>0.95:
	if (pos[count][0]>200 or pos[count][0]<160)and(pos[count][1]<140 or pos[count][1]>180):
	
            print w,pos[count]
	    f.write(str(w))
	    f.write(str(pos[count])+'\r\n')

	    totalproblem += 1
    count += 1
f.write(str(totalproblem))
f.close()    
iniw /= sum(iniw)
print totalproblem
