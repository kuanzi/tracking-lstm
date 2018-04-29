 # -*- coding: utf-8 -*- 
import sys
import hog
import cv2
import numpy as np
import sample


raw=16
col=32

img_dir='./balltrain/test.jpg'
image=cv2.imread(img_dir,cv2.IMREAD_GRAYSCALE)
feature=sample.HOG_feature(image)
res=[]
res.append(feature)
feature_2=np.array(res)
print (image.shape)
print (feature_2.shape)
'''
example:
feature_2=[ [image] [image] [image] ]
     
     [image]=[ [+++++]
               [+++++]
               [+++++]
                      ]
feature_2.shape=(1,9,32)
'''
import cnc_rnn
prediction=cnc_rnn.predict(feature_2)
print (prediction)
print (prediction[0][0])
