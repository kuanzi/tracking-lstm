# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 11:39:33 2017

@author: Administrator

通过多个点的坐标重心建立mask重心，去掉远离点并重建重心
"""

import numpy as np
import cv2

import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#计算初始mask重心
def mask_center(p1r):
    total_x = 0
    total_y = 0
    n = len(p1r)
    for i in p1r:
        total_x += i[0][0]
        total_y += i[0][1]
    center0x, center0y = total_x / n, total_y / n
    return center0x, center0y

def mask_center2(p1r):
    total_x = 0
    total_y = 0
    n = len(p1r)
    for i in p1r:
        total_x += i[0]
        total_y += i[1]
    center0x, center0y = total_x / n, total_y / n
    return center0x, center0y

def mask_create(img, p1r, boxh, boxw):
    center0x, center0y = mask_center(p1r)
    #判断点是否远离mask边界
    origin_len = len(p1r)
    for num in range(origin_len):
        boundup = center0x - boxh/2
        bounddown = center0x + boxh/2
        boundleft = center0y - boxw/2
        boundright = center0y + boxw/2
        
        new_p1r = []
        for i in p1r:
            if boundup < i[0][0] < bounddown:
                if boundleft < i[0][1] < boundright:
                    new_p1r.append(i)
                    
        flag = len(new_p1r)/len(p1r)
        
        
        if flag < 0.95:
            #求出距离最远的点，并剔除
            count = 0
            dmax = 0
            idmax = 0
            for i in p1r:
                
                d = ((i[0][0] - center0x)**2 + (i[0][1] - center0y)**2)**0.5
                if d > dmax:
                    dmax = d
                    idmax = count
                count = count + 1
            del(p1r[idmax])
            center0x, center0y = mask_center(p1r)
    
        else:
            break
    #建立新的mask
    mask = np.zeros_like(img)#初始化和视频大小相同的图像  
    #mask[:] = 255#将mask赋值255也就是算全部图像的角点
    up = center0y - boxh/2
    down = center0y + boxh/2
    left = center0x - boxw/2
    right = center0x + boxw/2
    up = int(up)
    down = int(down)
    left = int(left)
    right = int(right)
    if up < 0:
        up = 0
    if down > img.shape[0]:
        down = int(img.shape[0])
    if left < 0:
        left =0
    if right > img.shape[1]:
        right = int(img.shape[0])
    mask[up:down, left:right] = 255
    
    return mask


def mask_create_pos(img, pos, boxh, boxw):
    center0x, center0y = pos[0],pos[1]
    #判断点是否远离mask边界

    #建立新的mask
    mask = np.zeros_like(img)#初始化和视频大小相同的图像  
    #mask[:] = 255#将mask赋值255也就是算全部图像的角点
    up = center0y - boxh/2
    down = center0y + boxh/2
    left = center0x - boxw/2
    right = center0x + boxw/2
    up = int(up)
    down = int(down)
    left = int(left)
    right = int(right)
    if up < 0:
        up = 0
    if down > img.shape[0]:
        down = int(img.shape[0])
    if left < 0:
        left =0
    if right > img.shape[1]:
        right = int(img.shape[0])
    mask[up:down, left:right] = 255
    
    return mask