# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 10:19:34 2017

@author: Administrator
"""
from __future__ import division 

import numpy as np
import cv2

import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import copy
import feature
import readimg
import new_mask
import output
flag = 30
#param
#path = 'C:/Users/runqing/Desktop/HOF_new/data/ball'
path = './Woman/img'


if 'DragonBaby' in path:
    pos = np.array([115, 188])
    boxh, boxw = [65,56]
    filepath = 'D:/visualbenchmarkdatabase/VTB1.0_DataSequence/DragonBaby/result/'
    redetect = 10
    
if 'Biker' in path:
    pos = np.array([107, 270])
    boxh, boxw = [16,26]
    filepath = 'D:/visualbenchmarkdatabase/VTB1.0_DataSequence/Biker/result/'
    redetect = 20
    
if 'ball' in path:
    pos = np.array([136, 223])
    boxh, boxw = [46,46]
    filepath = 'C:/Users/runqing/Desktop/HOF_new/boundingresult1/'
    redetect = 20
    
if 'person' in path:
    pos = np.array([122, 158])
    boxh, boxw = [130,40]
    filepath = 'C:/Users/runqing/Desktop/HOF_new/boundingresult2/'
    redetect = 20
    
if 'juice' in path:
    pos = np.array([94, 149])
    boxh, boxw = [88,36]
    filepath = 'C:/Users/runqing/Desktop/HOF_new/boundingresult3/'
    redetect = 20
    
    
if 'Bolt' in path:
    pos = np.array([195, 349])
    boxh, boxw = [61,26]
    filepath = 'D:/visualbenchmarkdatabase/VTB1.0_DataSequence/Bolt/'
    redetect = 20
    
if 'Car1' in path:
    pos = np.array([115, 56])
    boxh, boxw = [66,55]
    filepath = 'D:/visualbenchmarkdatabase/VTB1.0_DataSequence/Car1/'
    redetect = 20

if 'CarDark' in path:
    pos = np.array([137, 87])
    boxh, boxw = [23,29]
    filepath = 'D:/visualbenchmarkdatabase/VTB1.0_DataSequence/CarDark/'
    redetect = 100
    
if 'CarScale' in path:
    pos = np.array([179, 27])
    boxh, boxw = [26,42]
    filepath = 'D:/visualbenchmarkdatabase/VTB1.0_DataSequence/CarScale/'
    redetect = 40

if 'Couple' in path:
    pos = np.array([78, 63])
    boxh, boxw = [62,25]
    filepath = 'D:/visualbenchmarkdatabase/VTB1.0_DataSequence/Couple/'
    redetect =39
    

if 'Woman' in path:
    pos = np.array([168, 223])
    boxh, boxw = [95,21]
    filepath = './Woman/'
    redetect =39
    
if 'BlurCar2' in path:
    pos = np.array([256, 280])
    boxh, boxw = [90,120]
    filepath = 'D:/visualbenchmarkdatabase/VTB1.0_DataSequence/BlurCar2/'
    
    redetect = 47
    flag = 30
    
if 'Walking' in path:
    pos = np.array([478, 704])
    boxh, boxw = [79,24]
    filepath = 'D:/visualbenchmarkdatabase/VTB1.0_DataSequence/Walking/'
    
    redetect = 19
    flag = 9
    
if 'Walking2' in path:
    pos = np.array([189, 149])
    boxh, boxw = [110,35]
    filepath = 'D:/visualbenchmarkdatabase/VTB1.0_DataSequence/Walking2/'
    
    redetect = 100
    flag = 9
    
if 'Surfer' in path:
    pos = np.array([150, 286])
    boxh, boxw = [26,23]
    filepath = 'D:/visualbenchmarkdatabase/VTB1.0_DataSequence/Surfer/'
    
    redetect = 20
    flag =10
if 'ClifBar' in path:
    pos = np.array([152, 158])
    boxh, boxw = [54,30]
    filepath = 'D:/visualbenchmarkdatabase/VTB1.0_DataSequence/ClifBar/'
    
    redetect = 15
    flag = 20
if 'Skating2' in path:
    pos = np.array([185, 321])
    boxh, boxw = [236,64]
    filepath = 'D:/visualbenchmarkdatabase/VTB1.0_DataSequence/Skating2/'
    
if 'Girl' in path:
    pos = np.array([43, 72])
    boxh, boxw = [45,31]
    filepath = 'D:/visualbenchmarkdatabase/VTB1.0_DataSequence/Girl/'
    redetect = 65

his_center = copy.deepcopy(pos)
groundtruth = []
import numpy as np
#f = open("C:/Users/runqing/Desktop/HOF_new/Vid_A_ball/groundtruth.txt")
f = open("./Woman/groundtruth_rect.txt")             # 返回一个文件对象  
#line = f.readline()             # 调用文件的 readline()方法  

for eachline in f.readlines():
        #print eachline
    lines = eachline.strip()
    lines = eachline.split(' ')
    lines = eachline.split('\t')
    lines[3] = lines[3].strip()
    print lines
    tem_pos = np.array([0,0])
    length = len(lines)
    for i in range(length):
        lines[i] = float(lines[i])
    tem_pos[0] = lines[1] + lines[3] / 2
    tem_pos[1] = lines[0] + lines[2] / 2
    line = f.readline()  
    
    groundtruth.append(tem_pos)

   
lk_params = dict( winSize  = (15, 15),   
                  maxLevel = 2,   
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))      
  
feature_params = dict( maxCorners = 500,   
                       qualityLevel = 0.3,  
                       minDistance = 7,  
                       blockSize = 7 )  
    
seq = readimg.readseq(path)
tempos = pos.copy()
out_seq = []
##############
#flow tracker#
##############
from skimage.color import rgb2gray
prevgray0 = rgb2gray(seq[0])
#prev0特征点检测
print(prevgray0.shape)
mask = np.zeros_like(prevgray0)#初始化和视频大小相同的图像  
#mask[:] = 255#将mask赋值255也就是算全部图像的角点
mask0 = int(tempos[0]-boxh/2)
mask1 = int(tempos[0]+boxh/2)
mask2 = int(tempos[1]-boxw/2)
mask3 = int(tempos[1]+boxw/2)
mask[mask0:mask1,mask2:mask3] = 255
p=[]#prevframe特征点
p = cv2.goodFeaturesToTrack(prevgray0, mask = mask, **feature_params)#像素级别角点检测
    
tracks = []#存放检测到的角点
if p is not None:  
    for x, y in np.float32(p).reshape(-1, 2):  
        tracks.append([(x, y)])#将检测到的角点放在待跟踪序列中
p.reshape(-1,2) 
#绘制prev以及其特征点
tem_mask = mask.copy()
tem_prev = prevgray0.copy()
tem_mask[:]=255                          
for x, y in [np.int32(tr[-1]) for tr in tracks]:#跟踪的角点画圆  
    cv2.circle(tem_mask, (x, y), 5, 0, -1)
for x, y in [np.int32(tr[-1]) for tr in tracks]:#跟踪的角点画圆  
    cv2.circle(tem_prev, (x, y), 5, 0, -1)
#plt.show()   
#plt.imshow(mask)
#plt.show()
#plt.imshow(tem_prev)


center_list = []
center_list.append(pos)


for i in range(len(seq)):
    
    j = i + 1#preframe角标
    if j == len(seq):
        print('tracking finish!')
        break
    
    gray = rgb2gray(seq[j])
    prevgray = rgb2gray(seq[i])
    vis = seq[j].copy()
    vis1 = seq[j].copy()
        
    #检测下一帧对应特征点
    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prevgray, gray, p0, None, **lk_params)#前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
    p0r, st, err = cv2.calcOpticalFlowPyrLK(gray, prevgray, p1, None, **lk_params)#当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置  
    d = abs(p0-p0r).reshape(-1, 2).max(-1)#得到角点回溯与前一帧实际角点的位置变化关系  
    good = d < flag#判断d内的值是否小于flag，大于1跟踪被认为是错误的跟踪点  
    #绘制下一帧对应特帧点
    his_p1 = p1
    point = []
    if p1 is not None:  
        for x, y in np.float32(p1).reshape(-1, 2):  
            point.append([(x, y)])#将检测到的角点放在待跟踪序列中
    for x, y in [np.int32(tr[-1]) for tr in point]:#跟踪的角点画圆  
        cv2.circle(gray, (x, y), 5, 0, -1)
    #plt.show()
    #plt.imshow(gray)
    
    
    #绘制点的运动轨迹
    new_tracks = []
    p1r = []
    for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):#将跟踪正确的点列入成功跟踪点  
        if not good_flag:
            continue
        tr.append((x, y))
        p1r.append([(x, y)])
        new_tracks.append(tr)  
        cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
    
    if p1r ==[]:
        for tr, (x, y) in zip(tracks, p1.reshape(-1, 2)):#将跟踪正确的点列入成功跟踪点  
            tr.append((x, y))
            p1r.append([(x, y)])
            new_tracks.append(tr)  
            cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
        
    tracks = copy.deepcopy(new_tracks)
    '''
    if j > 10:
        rad = 5
        [frame_pos_list, frame_feature_list] = feature.feature_create(p1r, seq, rad, j, boxw, boxh)#选定特征点的所在的周围100个位置的连续10帧patch的特征合并作为每个位置的特征
    '''
    bounding_x = 0
    bounding_y = 0
    length = len(p1r)
    for feature_point in p1r:
        bounding_x += feature_point[0][0]
        bounding_y += feature_point[0][1]
    bounding_x = bounding_x / length
    bounding_y = bounding_y / length
    frame_center = np.array([0, 0])
    frame_center[0] = bounding_x
    frame_center[1] = bounding_y
    
    his_center = frame_center
    
    center_list.append(frame_center)
    
    left_coner = frame_center - boxh/2
    right_coner = frame_center + boxh/2
    #bounding-box
    cv2.rectangle(vis1,(int(bounding_x - boxw/2),int(bounding_y - boxh/2)),(int(bounding_x + boxw/2),int(bounding_y + boxh/2)),(0,255,0),2)
    #point
    cv2.polylines(vis, [np.int32(tr) for tr in new_tracks], False, (0, 255, 0))#以上一振角点为初始点，当前帧跟踪到的点为终点划线  
    #plt.show()
    #plt.imshow(vis1)
    
    #plt.show()
    #plt.imshow(vis)
    print i
    
    out_seq.append(vis1)

    if p1r == []:
        print('object is missed')
        break
    #每五帧重新检测一次特征点
    if j % redetect == 0:
        
        mask = new_mask.mask_create(prevgray0, p1r, boxh, boxw)
        p=[]#prevframe特征点
        p = cv2.goodFeaturesToTrack(gray, mask = mask, **feature_params)#像素级别角点检测
            
        tracks = []#存放检测到的角点
        if p is not None:  
            for x, y in np.float32(p).reshape(-1, 2):  
                tracks.append([(x, y)])#将检测到的角点放在待跟踪序列中
                
                             
#############
#save output#
#############

output.save_out(filepath, out_seq)

import math

error_list = []
for i in range(len(center_list)):
    
    error0 = (abs(groundtruth[i][1] - center_list[i][0]))
    error1 = (abs(groundtruth[i][0] - center_list[i][1]))
    error = error0*error0 + error1*error1
    error = math.sqrt(error)
    error_list.append(error)
        
i = 0
for m in error_list:
    if m<20:
        i = i+1

success_rate = i/len(error_list)
                             
