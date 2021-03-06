# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:38:45 2017

This is PF for RNN
@author: zhang
"""

from numpy import *  
from numpy.random import *  
###################HOG提取函数
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import math
import hog

print('start')    

'''
重采样resample
'''
def resample(weights):  
    n = len(weights)  
    indices = []  
    # 求出离散累积密度函数(CDF)  
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]   
    # 选定一个随机初始点  
    u0, j = random(), 0  
    for u in [(u0+i)/n for i in range(n)]: # u 线性增长到 1  
        while u > C[j]: # 碰到小粒子，跳过  
            j+=1  
        indices.append(j-1)  # 碰到大粒子，添加，u 增大，还有第二次被添加的可能  
    return indices # 返回大粒子的下标
    
'''
切图片crop
'''

def croppatch( img, pos, boxw, boxh):
        tem_patch = img[pos[0] - boxh/2:pos[0] + boxh/2, pos[1] - boxw/2:pos[1] + boxw/2]
        return tem_patch

'''
粒子滤波初始化
'''
frame0 = cv2.imread('../Jumping/img/0001.jpg', cv2.IMREAD_GRAYSCALE)#读入第一帧图片
#frame0 = np.sqrt(frame0 / float(np.max(frame0)))#gamma滤波

stepsize = 8#粒子宽度
n = 100#粒子数目

pos0 = array([127, 163]) #第一帧目标中心
#pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2); pos的计算方法
boxw=34
boxh=33#box-bounding长宽

patch_target = croppatch(frame0, pos0, boxw, boxh)#目标patch切取
#imshow(patch_target)       
#import HOG_for_RNN

f0 = hog.HOG_feature(patch_target)#目标HOG特征提取

x = ones((n, 2), int) * pos0  #100个粒子初始中心   

'''读取图片序列'''
import os

seq = []
for filename in os.listdir('../Jumping/img'):   #listdir的参数是文件夹的路径
    #print ( filename)                                  #此时的filename是文件夹中文件的名称
    path_tem = '../Jumping/img/'+filename      #每一帧图片路径
    print(path_tem)	
    tem_frame = cv2.imread(path_tem, cv2.IMREAD_GRAYSCALE)#读取每一帧图片
    #tem_frame = np.sqrt(tem_frame / float(np.max(tem_frame)))#gamma滤波
    #plt.imshow(tem_frame)
    seq.append(tem_frame)
    
'''
每一帧的粒子滤波
'''
i=0#错误定位坐标：帧
patch_database = []
pos_database = []
feature_database = []
w_database = []
xpre = []
ypre = []

import cnc_rnn
import gc
imname = 0

for im in seq:
    i = i+1
    patch_data = []
    pos_data = []
    feature_data = []
    w_data = []
    
    wxim = []
    wyim = []
    #print(i)
    # 在上一帧的粒子周围撒点, 作为当前帧的粒子  
    m=uniform(-stepsize, stepsize, x.shape).transpose() 
    m=int32(m).transpose()
    x += m
    # 去掉超出画面边框的粒子  
    x  = x.clip(zeros(2), array(im.shape)-1).astype(int)
    '''
    得到每个粒子的patch以及HOG特征
    '''
    j = 0#错误定位坐标：粒子
    iniw = zeros(n)#初始化某一帧的权值iniw
    raw = 9
    col = 32
    for pos_tem in x:
        j = j+1
        #print(pos_tem)
        patch_particle = croppatch(im, pos_tem, boxw, boxh)#得到每个粒子的patch
        if patch_particle.shape[0]!=0:#无效patch剔除
        
            try:
                feature_particle = hog.HOG_feature(patch_particle)#得到每个patch的HOG
		#print(feature_particle)
                feature_particle = np.array(feature_particle)
                pos_data.append(pos_tem)
                feature_data.append(feature_particle)
                
                patch_data.append(patch_particle)#存放每一帧patch的34*32HOG
                
                count = j-1#有效粒子计数
                
                
                
               
            except:
                print(i)
                print(j)
                pass
    
    #计算当前im期望位置prex prey，归一化权重
    
    data = np.array(feature_data)
    print(data.shape)
    print(i)
    print(len(feature_particle))
    
    try:
        tem_w=cnc_rnn.predict(data)
    except:
        print(data)
        print(data.shape)
        print(type(feature_particle))
        for each_one in data:
            print(type(each_one))
        break
    #break


#for each_one in tem_w:
    #print(each_one)

'''    
    wsum = 0
    for each_w in w_data:
        wsum = wsum + each_w
        
    prex = 0
    for each_wx in wxim:
        prex = prex + int(each_wx)
    prex = prex/wsum  
    
    prey = 0
    for each_wy in wyim:
        prey = prey + int(each_wy)
    prey = prey/wsum
    
    xpre.append(prex)#存放每一个im的预测坐标x
    ypre.append(prey)#存放每一个im的预测坐标y
    
    patch_database.append(patch_data)#存放313帧的patch
    pos_database.append(pos_data)#存放313帧patch的有效pos
    feature_database.append(feature_data)#存放313帧有效patch的HOG
    print('ok')
    
    w_database.append(w_data)#存放313朕有效patch的权值
    
    p1=[]
    p2=[]
    py_tem1 = prey - boxw/2
    px_tem1 = prex - boxh/2
    py_tem1 = int(py_tem1)
    px_tem1 = int(px_tem1)

    p1 = [py_tem1,px_tem1]
    p1 = tuple(p1)

    py_tem2 = prey + boxw/2
    px_tem2 = prex + boxh/2
    py_tem2 = int(py_tem2)
    px_tem2 = int(px_tem2)

    p2 = [py_tem2,px_tem2]
    p2 = tuple(p2)
    #imname =+1
    #print(imname)
    if imname < 99:
        path_tem = '../Jumping/result/test00' + str(i) + '.png'
    elif imname < 999:
        path_tem = '../Jumping/result/test0' + str(i) + '.png'

    print p1,p2
    cv2.rectangle(im,p1,p2,(0,255,0),2)
    plt.imsave(path_tem,im)
'''    


