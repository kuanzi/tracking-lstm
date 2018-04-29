# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:55:49 2017

@author: runqing
"""
import numpy as np



def load_dataset_gt(gt_file):
   txtfile = open(gt_file, "r")
   lines = txtfile.read().split('\n')  #'\r\n'
   return lines

def find_gt_location(lines, id):
    #print("lines length: ", len(lines))
    #print("id: ", id)
    line = lines[id]
    elems = line.split('\t')   # for gt type 2
    #print(elems)
    if len(elems) < 4:
        elems = line.split(',') #for gt type 1
    #print(elems)
    x1 = elems[0]
    y1 = elems[1]
    w = elems[2]
    h = elems[3]
    gt_location = [int(x1), int(y1), int(w), int(h)]
    return gt_location

def find_gt_location_onlyxy(lines, id):
    #print("lines length: ", len(lines))
    #print("id: ", id)
    line = lines[id]
    elems = line.split('\t')   # for gt type 2
    #print(elems)
    if len(elems) < 4:
        elems = line.split(',') #for gt type 1
    #print(elems)
    x1 = elems[0]
    y1 = elems[1]
    gt_location = [int(x1), int(y1)]
    return gt_location

def extension_location(gt_0,gt_1):
    gt_3 = [0,0,0,0]
    gt_3[0] = gt_1[0] + gt_1[0] - gt_0[0]
    gt_3[1] = gt_1[1] + gt_1[1] - gt_0[1]
    gt_3[2] = int((gt_0[2]+gt_1[2])/2)
    gt_3[3] = int((gt_0[3]+gt_1[3])/2)
    return gt_3
    
def extension_location_onlyxy(gt_0,gt_1):
    gt_3 = [0,0]
    gt_3[0] = gt_1[0] + gt_1[0] - gt_0[0]
    gt_3[1] = gt_1[1] + gt_1[1] - gt_0[1]
    return gt_3

def extension_groundtruth(path):
    extension_gt = []
    lines = load_dataset_gt(path)
    for m in lines:
        if m == '':
            lines.remove(m)
    extension_gt.append(find_gt_location(lines,0))
    extension_gt.append(find_gt_location(lines,1))
    for id in range(len(lines)):
        id_new = id + 1
        if (id_new + 1) == len(lines):
            break
        location_0 = find_gt_location(lines,id)
        location_1 = find_gt_location(lines,id_new)
        extension = extension_location(location_0,location_1)
        extension_gt.append(extension)
    return extension_gt

def extension_groundtruth_onlyxy(path):
    extension_gt = []
    lines = load_dataset_gt(path)
    for m in lines:
        if m == '':
            lines.remove(m)
    extension_gt.append(find_gt_location_onlyxy(lines,0))
    extension_gt.append(find_gt_location_onlyxy(lines,1))
    for id in range(len(lines)):
        id_new = id + 1
        if (id_new + 1) == len(lines):
            break
        location_0 = find_gt_location(lines,id)
        location_1 = find_gt_location(lines,id_new)
        extension = extension_location_onlyxy(location_0,location_1)
        extension_gt.append(extension)
    return extension_gt

def make_gt(path):
    gt = []
    lines = load_dataset_gt(path)
    for m in lines:
        if m == '':
            lines.remove(m)
    for id in range(len(lines)):
        location = find_gt_location(lines,id)
        gt.append(location)
    return gt        

def make_gt_onlyxy(path):
    gt = []
    lines = load_dataset_gt(path)
    for m in lines:
        if m == '':
            lines.remove(m)
    for id in range(len(lines)):
        location = find_gt_location_onlyxy(lines,id)
        gt.append(location)
    return gt

def make_input(train_data,train_label):
    data = []
    label = []
    import copy
    data,label = copy.deepcopy(train_data[0]),copy.deepcopy(train_label[0])
    if len(train_data) == len(train_label):
        for id in range(len(train_data)):
            id_new = id + 1
            data.extend(train_data[id_new])
            label.extend(train_label[id_new])
            if (id_new+1) == len(train_data):
                break
    else:
        print('extension does not match groundtruth')
    data = np.reshape(data,(len(data),1,4))
    label = np.reshape(label,(len(label),4))
    return data,label

def make_input_onlyxy(train_data,train_label):
    data = []
    label = []
    import copy
    data,label = copy.deepcopy(train_data[0]),copy.deepcopy(train_label[0])
    if len(train_data) == len(train_label):
        for id in range(len(train_data)):
            id_new = id + 1
            data.extend(train_data[id_new])
            label.extend(train_label[id_new])
            if (id_new+1) == len(train_data):
                break
    else:
        print('extension does not match groundtruth')
    data = np.reshape(data,(len(data),1,2))
    label = np.reshape(label,(len(label),2))
    return data,label

def main_data():
    name_list = ['Bird1','BlurBody','BlurCar1','BlurCar3','BlurCar4','Boy','Car1','Car4','CarDark',\
             'CarScale','Couple','Dancer','Dancer2','David3','Diving','Dog','DragonBaby','Girl2',\
             'Gym','Human2','Human3','Human4','Human6','Human7','Human8','Human9','Jogging_1',\
             'Jogging_2','Jump','Singer1','Singer2','Skater','Skater2','Skating1','Skiing','Surfer',\
             'Suv','Trans','Walking2','Woman']
    train_data = []
    train_label = []       
    for i in range(len(name_list)):
        gt_path = './region/'+name_list[i]+'/groundtruth_rect.txt'
        data = extension_groundtruth(gt_path)
        label = make_gt(gt_path)
        train_data.append(data)
        train_label.append(label)
    data,label = make_input(train_data,train_label)
    return data,label

def main_data_onlyxy():
    name_list = ['Bird1','BlurBody','BlurCar1','BlurCar3','BlurCar4','Boy','Car1','Car4','CarDark',\
             'CarScale','Couple','Dancer','Dancer2','David3','Diving','Dog','DragonBaby','Girl2',\
             'Gym','Human2','Human3','Human4','Human6','Human7','Human8','Human9','Jogging_1',\
             'Jogging_2','Jump','Singer1','Singer2','Skater','Skater2','Skating1','Skiing','Surfer',\
             'Suv','Trans','Walking2','Woman']
    train_data = []
    train_label = []       
    for i in range(len(name_list)):
        gt_path = './region/'+name_list[i]+'/groundtruth_rect.txt'
        data = extension_groundtruth_onlyxy(gt_path)
        label = make_gt_onlyxy(gt_path)
        train_data.append(data)
        train_label.append(label)
    data,label = make_input_onlyxy(train_data,train_label)
    return data,label

def test_data():
    name_list = ['RedTeam','Panda','Ironman','David']
    train_data = []
    train_label = []       
    for i in range(len(name_list)):
        gt_path = './test/'+name_list[i]+'/groundtruth_rect.txt'
        data = extension_groundtruth(gt_path)
        label = make_gt(gt_path)
        train_data.append(data)
        train_label.append(label)
    data,label = make_input(train_data,train_label)
    return data,label


def test_data_onlyxy():
    name_list = ['RedTeam','Panda','Ironman','David']
    train_data = []
    train_label = []       
    for i in range(len(name_list)):
        gt_path = './test/'+name_list[i]+'/groundtruth_rect.txt'
        data = extension_groundtruth_onlyxy(gt_path)
        label = make_gt_onlyxy(gt_path)
        train_data.append(data)
        train_label.append(label)
    data,label = make_input_onlyxy(train_data,train_label)
    return data,label
