# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 17:28:38 2017

@author: Administrator
"""

import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import copy


import readimg

def getdata():
    #param
    path = './ball/'
    step = 4#窗口移动步数
    
    if 'ball' in path:
        pos = np.array([136, 223])
        boxh, boxw = [46,46]
        
        
    seq = readimg.readseq(path)
    tempos = pos.copy()
    
    def draw_flow(img, flow, step=16):
    
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = copy.deepcopy(img)	
        #vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis
    
    def calc_hist(flow):
    
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees = 1)
        
        q1 = float(((0 < ang) & (ang <= 45)).sum())
        q2 = float(((45 < ang) & (ang <= 90)).sum())
        q3 = float(((90 < ang) & (ang <= 135)).sum())
        q4 = float(((135 < ang) & (ang <= 180)).sum())
        q5 = float(((180 < ang) & (ang <= 225)).sum())
        q6 = float(((225 <= ang) & (ang <= 270)).sum())
        q7 = float(((270 < ang) & (ang <= 315)).sum())
        q8 = float(((315 < ang) & (ang <= 360)).sum())    
        hist = [q1, q2, q3, q4 ,q5, q6, q7 ,q8]
        
        return (hist)
    
    def nor_hist(hist):
        new_hist = []
        for each_one in hist:
            tem = (each_one - min(hist))/(max(hist)-min(hist))
            new_hist.append(tem)
            
        return (new_hist)
    
    #每帧图片各个窗口的中心坐标
    def win_center(gray_img, step, boxh, boxw):#step为窗口步数
        row, col = int(gray_img.shape[0]), int(gray_img.shape[1])
        pos_list = []
        cen_pos = np.array([boxh/2, boxw/2])
        rad_row = (row - boxh) / step + 1
        rad_col = (col - boxw) / step + 1
    
        basepos = copy.deepcopy(cen_pos)
        basepos = basepos - step
        for i in range(rad_row):
            basepos[0] += step
            varpos = copy.deepcopy(basepos)
            for j in range(rad_col):
                varpos[1] += step
                tem_pos = copy.deepcopy(varpos)
                pos = np.array(tem_pos)
                pos_list.append(pos)
        return pos_list
    
    #提取每一帧每一个window特征
    len_seq = len(seq)
    
    prevgray = copy.deepcopy(seq[0])
    #prevgray = cv2.cvtColor(seq[0],cv2.COLOR_BGR2GRAY)#初始化prevgray
    pos_list = win_center(prevgray, step, boxh, boxw)
    
    seq_map = []
    
    for i in range(len_seq):
        frame_map = []
        j = i + 1
        if j == 602:
            break
        prevgray = copy.deepcopy(seq[i])
        gray = copy.deepcopy(seq[j])
        #prevgray = cv2.cvtColor(seq[i],cv2.COLOR_BGR2GRAY)#初始化上一帧prevgray
        #gray = cv2.cvtColor(seq[j],cv2.COLOR_BGR2GRAY)#初始化当前帧gray
        for pos in pos_list:
            imgA = readimg.croppatch(prevgray, pos, boxw, boxh)
            imgB = readimg.croppatch(gray, pos, boxw, boxh)
            flow = cv2.calcOpticalFlowFarneback(imgA, imgB, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            patch = draw_flow(imgB,flow)
            temhist = calc_hist(flow)
            hist = nor_hist(temhist)
            histgram = copy.deepcopy(hist)
            histgram = np.array(histgram)
            frame_map.append(histgram)
        seq_map.append(frame_map)
        print(i)
        
    #制作时序特征
    feature_length = 10#特征长度，即每个特征矩阵包含的帧数-1
    
    T0 = range(feature_length - 1, len(seq_map))
    T0 = np.array(T0)
    T = T0 - feature_length + 1#时间标签
    
    feature_map = []
    p = 0
    for t0 in T:
        tstart = t0
        tend = t0 + feature_length
        #t=0,t0=10,即第11帧时
        feature_map_t = []#t时刻的feature_map
        for count in range(len(pos_list)):
            tem_feature = np.array([])
            for t in range(tstart, tend):
                tem_feature = np.append(tem_feature, seq_map[t][count])
            feature_map_t.append(tem_feature)
        if p == 0:
            feature = feature_map_t
        else:
            feature = np.r_[feature,feature_map_t]
        feature_map.append(feature_map_t)
        p = p + 1

    #train
    negative_feature = feature[0:1725,:].reshape(1725,1,80)
    
    #positive
    positive_feature = []
    tem_posi = feature[2050,:].reshape(1,80)
    for i in range(1725):
        positive_feature.append(tem_posi)
    positive_feature = np.array(positive_feature)
    feature_train = np.r_[positive_feature,negative_feature]
    
    #test
    #train
    negative_feature = feature[1725:2025,:].reshape(300,1,80)
    
    #positive
    positive_feature = []
    tem_posi = feature[5431,:].reshape(1,80)
    for i in range(300):
        positive_feature.append(tem_posi)
    positive_feature = np.array(positive_feature)
    feature_test = np.r_[positive_feature,negative_feature]
    
    
    #train data
    label_positive = []
    label_negative = []
    label = []
    for i in range(1725):
        label_pos = np.array([0,1])
        label_positive.append(label_pos)
        label_neg = np.array([1,0])
        label_negative.append(label_neg)
            
    for i in range(1725):
        label_pos = np.array([0,1])
        label.append(label_pos)
    for i in range(1725):
        label_pos = np.array([1,0])
        label.append(label_neg)     
    label = np.array(label)
    #test data
    label_test = []
    for i in range(300):
        label_pos = np.array([0,1])
        label_test.append(label_pos)
    for i in range(300):
        label_pos = np.array([1,0])
        label_test.append(label_neg)     
    label_test = np.array(label_test)
    return feature_train,label,feature_test,label_test



