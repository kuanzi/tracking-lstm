# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 21:33:45 2017

@author: runqing
"""
def getdata():
    import readimg
    
    positive = []
    negative = []
    
    test = []
    test1 = []
    test2 = []
    path_1 = './train/negative/'
    path_2 = './train/positive/'
    path_3 = './train/test_positive/'
    path_4 = './train/test_negative/'
    
    positive = readimg.readseq(path_2)
    negative = readimg.readseq(path_1)
    test1 = readimg.readseq(path_3)
    test2 = readimg.readseq(path_4)
    test1.extend(test2)
    test = test1
    
    import numpy as np
    import HOG
    import math
    import matplotlib.pyplot as plt
    #train data
    label_positive = []
    label_negative = []
    label = []
    for i in range(len(positive)):
        label_pos = np.array([0,1])
        label_positive.append(label_pos)
        label_neg = np.array([1,0])
        label_negative.append(label_neg)
        
    for i in range(len(positive)):
        label_pos = np.array([0,1])
        label.append(label_pos)
    for i in range(len(negative)):
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
    
    #hogfeature train
    hogfeature = []
    for img in positive:    
        hog = HOG.Hog_descriptor(img, cell_size=8, bin_size=8)
        vector, image = hog.extract()
        vector = np.array(vector)
        vector = vector.reshape((1,512))
        hogfeature.append(vector)
        #print np.array(vector).shape
        #plt.imshow(image, cmap=plt.cm.gray)
        #plt.show()
    for img in negative:    
        hog = HOG.Hog_descriptor(img, cell_size=8, bin_size=8)
        vector, image = hog.extract()
        vector = np.array(vector)
        vector = vector.reshape((1,512))
        hogfeature.append(vector)
        #print np.array(vector).shape
    hogfeature = np.array(hogfeature)
    #hogfeature test
    test_feature = []
    for img in test:    
        hog = HOG.Hog_descriptor(img, cell_size=8, bin_size=8)
        vector, image = hog.extract()
        vector = np.array(vector)
        vector = vector.reshape((1,512))
        test_feature.append(vector)
    test_feature = np.array(test_feature)
    return hogfeature,label,test_feature,label_test
