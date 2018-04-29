# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:05:25 2017

@author: Administrator
"""

'''
This is HOG for RNN patch,length of each feature is 2*2*8,quatity of features is
(imgweight/cell_size-1)*(imgheight/cell_size-1)
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt 
import math



def HOG_feature(img):
    
    '''
    统计cell的方向直方图的函数cell_gradient，属于该方向时，该hist的bin+1
    '''
    def cell_gradient(cell_magnitude, cell_angle):
        orientation_centers = [0] * bin_size
        for k in range(cell_magnitude.shape[0]):#shape[0]读取第一维的长度
            for l in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[k][l]
                gradient_angle = cell_angle[k][l]
                min_angle = int(gradient_angle / angle_unit)%8
                max_angle = (min_angle + 1) % bin_size
                mod = gradient_angle % angle_unit
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
        return orientation_centers
    
    img = np.sqrt(img / float(np.max(img)))                 #读入patch并gamma校正
    
    '''
    计算patch的梯度方向和大小
    '''
    height, width = img.shape
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
    gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
    
    '''
    将patch分为8*8个像素的cell
    统计梯度方向的直方图的bin的数量为8
    得到梯度方向直方图cell_gradient_vector
    '''
    cell_size = 8
    bin_size = 8
    angle_unit = 360 / bin_size
    gradient_magnitude = abs(gradient_magnitude)
    cell_gradient_vector = np.zeros((height / cell_size, width / cell_size, bin_size))#初始化8块切块
    #print(cell_gradient_vector)
    
    #print cell_gradient_vector.shape
    
    
    
    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
            cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
            #print cell_angle.max()
    
            cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)
            
    '''
    绘制特征图hog_image
    '''
    import math
    import matplotlib.pyplot as plt
    
    hog_image= np.zeros([height, width])
    cell_gradient = cell_gradient_vector
    cell_width = cell_size / 2
    max_mag = np.array(cell_gradient).max()
    for x in range(cell_gradient.shape[0]):
        for y in range(cell_gradient.shape[1]):
            cell_grad = cell_gradient[x][y]
            cell_grad /= max_mag
            angle = 0
            angle_gap = angle_unit
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                cv2.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                angle += angle_gap
    
    #plt.imshow(hog_image, cmap=plt.cm.gray)
    #plt.show()
            
            
    '''
    得到HOG特征，每个block的HOG特征长度为2*2*8（一个block2*2个cell，每个cell有8个hist）
    '''
    hog_vector = []
    for i in range(cell_gradient_vector.shape[0] - 1):
        for j in range(cell_gradient_vector.shape[1] - 1):
            block_vector = []
            block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            hog_vector.append(block_vector)
    
    #print np.array(hog_vector).shape
    return hog_vector

'''
test
'''
#img = cv2.imread('person_037.jpg', cv2.IMREAD_GRAYSCALE)
#result=HOG_feature(img)
#print np.array(hog_vector).shape

                  
'''
img = np.sqrt(img / float(np.max(img)))                 #gamma校正

'''
'''
计算patch的梯度方向和大小
'''
'''
height, width = img.shape
gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)

'''
'''
将patch分为8*8个像素的cell
统计梯度方向的直方图的bin的数量为8
得到梯度方向直方图cell_gradient_vector
'''
'''
cell_size = 8
bin_size = 8
angle_unit = 360 / bin_size
gradient_magnitude = abs(gradient_magnitude)
cell_gradient_vector = np.zeros((height / cell_size, width / cell_size, bin_size))#初始化8块切块
#print(cell_gradient_vector)

print cell_gradient_vector.shape



for i in range(cell_gradient_vector.shape[0]):
    for j in range(cell_gradient_vector.shape[1]):
        cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
        cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                     j * cell_size:(j + 1) * cell_size]
        #print cell_angle.max()

        cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)
        
'''
#绘制特征图hog_image
'''
import math
import matplotlib.pyplot as plt

hog_image= np.zeros([height, width])
cell_gradient = cell_gradient_vector
cell_width = cell_size / 2
max_mag = np.array(cell_gradient).max()
for x in range(cell_gradient.shape[0]):
    for y in range(cell_gradient.shape[1]):
        cell_grad = cell_gradient[x][y]
        cell_grad /= max_mag
        angle = 0
        angle_gap = angle_unit
        for magnitude in cell_grad:
            angle_radian = math.radians(angle)
            x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
            y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
            x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
            y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
            cv2.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
            angle += angle_gap

#plt.imshow(hog_image, cmap=plt.cm.gray)
#plt.show()
        
        
'''
#得到HOG特征，每个block的HOG特征长度为2*2*8（一个block2*2个cell，每个cell有8个hist）
'''
hog_vector = []
for i in range(cell_gradient_vector.shape[0] - 1):
    for j in range(cell_gradient_vector.shape[1] - 1):
        block_vector = []
        block_vector.extend(cell_gradient_vector[i][j])
        block_vector.extend(cell_gradient_vector[i][j + 1])
        block_vector.extend(cell_gradient_vector[i + 1][j])
        block_vector.extend(cell_gradient_vector[i + 1][j + 1])
        mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
        magnitude = mag(block_vector)
        if magnitude != 0:
            normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
            block_vector = normalize(block_vector, magnitude)
        hog_vector.append(block_vector)

print np.array(hog_vector).shape
'''
