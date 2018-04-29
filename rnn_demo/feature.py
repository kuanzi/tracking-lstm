# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 11:05:27 2017

@author: Administrator
"""

import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import copy


import readimg

def draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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
    try:
        for each_one in hist:
            tem = (each_one - min(hist))/(max(hist)-min(hist))
            new_hist.append(tem)
    except:
        new_hist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return (new_hist)

def search_patch(cen_pos, rad):
    pos_list = []
    cen_pos[0], cen_pos[1] = int(cen_pos[0]) - rad, int(cen_pos[1]) - rad
    basepos = copy.deepcopy(cen_pos)
    dourad = 2 * rad
    for i in range(dourad):
        basepos[0] += 1
        varpos = copy.deepcopy(basepos)
        for j in range(dourad):
            varpos[1] += 1
            tem_pos = copy.deepcopy(varpos)
            pos_list.append(tem_pos)
    return pos_list
    

#以每个特征点为中心搜索patch

rad = 5#搜索半径

def feature_create(p1r, seq, rad, t, boxw, boxh):
    p1r_pos_list = []
    p1r_pos_list_feature = []
    
    [maxh, maxw, maxd] = seq[0].shape
    
    for i in p1r:
        cen_pos = np.array(i[0])

        pos_list = search_patch(cen_pos, rad)
        
        p1r_pos_list.append(pos_list)
        
        pos_list_feature = []
        
        for tem_pos in pos_list:
            crop_img0 = readimg.croppatch(seq[t], tem_pos, boxw, boxh)#当前帧patch
            crop_img0 = cv2.cvtColor(crop_img0,cv2.COLOR_BGR2GRAY)
            
            crop_img1 = readimg.croppatch(seq[t-1], tem_pos, boxw, boxh)
            crop_img1 = cv2.cvtColor(crop_img1,cv2.COLOR_BGR2GRAY)
            
            crop_img2 = readimg.croppatch(seq[t-2], tem_pos, boxw, boxh)
            crop_img2 = cv2.cvtColor(crop_img2,cv2.COLOR_BGR2GRAY)
            
            crop_img3 = readimg.croppatch(seq[t-3], tem_pos, boxw, boxh)
            crop_img3 = cv2.cvtColor(crop_img3,cv2.COLOR_BGR2GRAY)
            
            crop_img4 = readimg.croppatch(seq[t-4], tem_pos, boxw, boxh)
            crop_img4 = cv2.cvtColor(crop_img4,cv2.COLOR_BGR2GRAY)
            
            crop_img5 = readimg.croppatch(seq[t-5], tem_pos, boxw, boxh)
            crop_img5 = cv2.cvtColor(crop_img5,cv2.COLOR_BGR2GRAY)
            
            crop_img6 = readimg.croppatch(seq[t-6], tem_pos, boxw, boxh)
            crop_img6 = cv2.cvtColor(crop_img6,cv2.COLOR_BGR2GRAY)
            
            crop_img7 = readimg.croppatch(seq[t-7], tem_pos, boxw, boxh)
            crop_img7 = cv2.cvtColor(crop_img7,cv2.COLOR_BGR2GRAY)
            
            crop_img8 = readimg.croppatch(seq[t-8], tem_pos, boxw, boxh)
            crop_img8 = cv2.cvtColor(crop_img8,cv2.COLOR_BGR2GRAY)
            
            crop_img9 = readimg.croppatch(seq[t-9], tem_pos, boxw, boxh)
            crop_img9 = cv2.cvtColor(crop_img9,cv2.COLOR_BGR2GRAY)
            
            crop_img10 = readimg.croppatch(seq[t-10], tem_pos, boxw, boxh)
            crop_img10 = cv2.cvtColor(crop_img10,cv2.COLOR_BGR2GRAY)
            
            
            flow0 = cv2.calcOpticalFlowFarneback(crop_img1, crop_img0, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow1 = cv2.calcOpticalFlowFarneback(crop_img2, crop_img1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow2 = cv2.calcOpticalFlowFarneback(crop_img3, crop_img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow3 = cv2.calcOpticalFlowFarneback(crop_img4, crop_img3, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow4 = cv2.calcOpticalFlowFarneback(crop_img5, crop_img4, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow5 = cv2.calcOpticalFlowFarneback(crop_img6, crop_img5, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow6 = cv2.calcOpticalFlowFarneback(crop_img7, crop_img6, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow7 = cv2.calcOpticalFlowFarneback(crop_img8, crop_img7, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow8 = cv2.calcOpticalFlowFarneback(crop_img9, crop_img8, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow9 = cv2.calcOpticalFlowFarneback(crop_img10, crop_img9, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            temhist0 = calc_hist(flow0)
            temhist1 = calc_hist(flow1)
            temhist2 = calc_hist(flow2)
            temhist3 = calc_hist(flow3)
            temhist4 = calc_hist(flow4)
            temhist5 = calc_hist(flow5)
            temhist6 = calc_hist(flow6)
            temhist7 = calc_hist(flow7)
            temhist8 = calc_hist(flow8)
            temhist9 = calc_hist(flow9)
            
            hist0 = nor_hist(temhist0)
            hist1 = nor_hist(temhist1)
            hist2 = nor_hist(temhist2)
            hist3 = nor_hist(temhist3)
            hist4 = nor_hist(temhist4)
            hist5 = nor_hist(temhist5)
            hist6 = nor_hist(temhist6)
            hist7 = nor_hist(temhist7)
            hist8 = nor_hist(temhist8)
            hist9 = nor_hist(temhist9)
    
            hist = hist0 + hist1 + hist2 + hist3 + hist4 + hist5 + hist6 + hist7 + hist8 + hist9
            histgram = copy.deepcopy(hist)
            histgram = np.array(histgram)
            
            pos_list_feature.append(histgram)
            
        p1r_pos_list_feature.append(pos_list_feature)
        
        return p1r_pos_list, p1r_pos_list_feature
            
        #检测到的patch提取特征