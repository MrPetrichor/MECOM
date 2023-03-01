import numpy as np
from numpy.random import randint
import torch
import random


def get_task_data(feature_image_train_100,feature_audio_train_100,feature_image_test_100,feature_audio_test_100,label_train_100,label_test_100,class_num):
    class_list=[]
    for i in range(class_num):
        x=randint(0, 100)
        while(x in class_list):
            x = randint(0, 100)
        class_list.append(x)
    tr100_len = feature_image_train_100.size()[0]
    te100_len = feature_image_test_100.size()[0]
    image_train_10_list=[]
    audio_train_10_list=[]
    image_test_10_list = []
    audio_test_10_list = []
    label_train_10_list=[]
    label_test_10_list=[]
    for i in range(tr100_len):
        if(label_train_100[i] in class_list):
            image_train_10_list.append(feature_image_train_100[i])
            audio_train_10_list.append(feature_audio_train_100[i])
            index = class_list.index(label_train_100[i])
            label_train_10_list.append(index)

    for i in range(te100_len):
        if(label_test_100[i] in class_list):
            image_test_10_list.append(feature_image_test_100[i])
            audio_test_10_list.append(feature_audio_test_100[i])
            index = class_list.index(label_test_100[i])
            label_test_10_list.append(index)


    image_train_10 = torch.zeros(size=(len(label_train_10_list),2048))
    audio_train_10 = torch.zeros(size=(len(label_train_10_list), 2048))
    image_test_10 = torch.zeros(size=(len(label_test_10_list), 2048))
    audio_test_10 = torch.zeros(size=(len(label_test_10_list), 2048))
    label_train_10 = torch.zeros(size=(len(label_train_10_list), 1)).to(torch.int64)
    label_test_10 = torch.zeros(size=(len(label_test_10_list), 1)).to(torch.int64)

    for i in range(len(label_train_10_list)):
        image_train_10[i] = image_train_10_list[i]
        audio_train_10[i] = audio_train_10_list[i]
        label_train_10[i] = label_train_10_list[i]
    for i in range(len(label_test_10_list)):
        image_test_10[i] = image_test_10_list[i]
        audio_test_10[i] = audio_test_10_list[i]
        label_test_10[i] = label_test_10_list[i]

    label_train_10 = torch.squeeze(label_train_10)
    label_test_10 = torch.squeeze(label_test_10)

    return image_train_10, audio_train_10, image_test_10, audio_test_10, label_train_10, label_test_10

def get_task_data_test(feature_image_train_100,feature_audio_train_100,feature_image_test_100,feature_audio_test_100,label_train_100,label_test_100,class_num):

    class_list=[1,2,3,4,5,6,7,8,9,10]



    tr100_len = feature_image_train_100.size()[0]
    te100_len = feature_image_test_100.size()[0]
    image_train_10_list=[]
    audio_train_10_list=[]
    image_test_10_list = []
    audio_test_10_list = []
    label_train_10_list=[]
    label_test_10_list=[]
    for i in range(tr100_len):
        if(label_train_100[i] in class_list):
            image_train_10_list.append(feature_image_train_100[i])
            audio_train_10_list.append(feature_audio_train_100[i])
            index = class_list.index(label_train_100[i])
            label_train_10_list.append(index)

    for i in range(te100_len):
        if(label_test_100[i] in class_list):
            image_test_10_list.append(feature_image_test_100[i])
            audio_test_10_list.append(feature_audio_test_100[i])
            index = class_list.index(label_test_100[i])
            label_test_10_list.append(index)


    image_train_10 = torch.zeros(size=(len(label_train_10_list),2048))
    audio_train_10 = torch.zeros(size=(len(label_train_10_list), 2048))
    image_test_10 = torch.zeros(size=(len(label_test_10_list), 2048))
    audio_test_10 = torch.zeros(size=(len(label_test_10_list), 2048))
    label_train_10 = torch.zeros(size=(len(label_train_10_list), 1)).to(torch.int64)
    label_test_10 = torch.zeros(size=(len(label_test_10_list), 1)).to(torch.int64)

    for i in range(len(label_train_10_list)):
        image_train_10[i] = image_train_10_list[i]
        audio_train_10[i] = audio_train_10_list[i]
        label_train_10[i] = label_train_10_list[i]
    for i in range(len(label_test_10_list)):
        image_test_10[i] = image_test_10_list[i]
        audio_test_10[i] = audio_test_10_list[i]
        label_test_10[i] = label_test_10_list[i]

    label_train_10 = torch.squeeze(label_train_10)
    label_test_10 = torch.squeeze(label_test_10)

    return image_train_10, audio_train_10, image_test_10, audio_test_10, label_train_10, label_test_10

def get_task_data_100(label_train_100,label_test_100):
    class_list=[]
    for i in range(100):
        class_list.append(i)
    random.shuffle(class_list)

    tr100_len = label_train_100.size()[0]
    te100_len = label_test_100.size()[0]

    for i in range(tr100_len):
        label_train_100[i] = class_list[label_train_100[i]]

    for i in range(te100_len):
        label_test_100[i] = class_list[label_test_100[i]]


    return label_train_100, label_test_100

def get_task_data_test_100(feature_image_train_100,feature_audio_train_100,feature_image_test_100,feature_audio_test_100,label_train_100,label_test_100):

    return feature_image_train_100,feature_audio_train_100,feature_image_test_100,feature_audio_test_100,label_train_100,label_test_100




