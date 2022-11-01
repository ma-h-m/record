#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 如果这一句报错请用https://blog.csdn.net/weixin_43977534/article/details/107752562

# MNIST_data指的是存放数据的文件夹路径，one_hot=True 为采用one_hot的编码方式编码标签
mnist = input_data.read_data_sets('../datasets/MNIST_data/', one_hot=True)
# load data
train_X = mnist.train.images
train_Y = mnist.train.labels
test_X = mnist.test.images
test_Y = mnist.test.labels
W = np.random.normal(0,0.001,size = (10,784))
reg_str = 0.01 # 超参
for x,y in zip(train_X,train_Y):
    x = np.array(x)
    sj = np.dot(W,x)
    tmp = sj - y + 1
    l = np.zeros(10)
    label = np.argmax(y)
    if label == np.argmax(sj):
        loss = 0
    else:
        for i in range(len(y)):
            if i != label:
                l = max(0,tmp[i])
        loss = sum(l) / len(l)
    L = loss + reg_str * sum( sum(W * W))



