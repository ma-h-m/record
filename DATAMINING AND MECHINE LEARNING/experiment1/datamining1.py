import math
import random
from ctypes import sizeof
from enum import auto

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.datasets._base import Bunch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def get_in_data(): # 读入数据，构建数据集、正则化并拆分
    ss = StandardScaler()
    ss2 = StandardScaler()
    df = pd.read_csv('train.csv')
    train_data = Bunch()
    train_data.data = ss.fit_transform(np.array(df.iloc[:,1:385]))
    #train_data.target = ss2.fit_transform(np.array(df.iloc[:,385:]))
    train_data.target = np.array(df.iloc[:,385:])
    train_data.feature_names = np.array(df.columns[1:385])
    train_data.target_names = [df.columns[385]]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(train_data.data,train_data.target,test_size=0.1) # 拆分数据集
    
    df = pd.read_csv('test.csv')
    test_data = Bunch()
    test_data.data = ss.transform(np.array(df.iloc[:,1:385]))
    #train_data.target = ss2.fit_transform(np.array(df.iloc[:,385:]))
    test_data.feature_names = np.array(df.columns[1:385])
    return train_data,x_train,x_test,y_train,y_test,test_data

def h(x,theta,theta0):      # 定义h_theta函数
    tmp = np.dot(x,theta[0]) + theta0
    return tmp

def init():
    theta = np.random.randn(1,384) # 随机初始化
    theta0 = random.random()
    return theta,theta0
# 使用梯度下降的线性回归
def linear_regression(alpha,beta,batch_size,train_input,train_target,theta,theta0):

    loss = []
    num = train_input.shape[0]
    cnt = 0


    tmp_rec = []
    for j in range(100):
        np.random.shuffle(train_input)
        for i in range(int(num / batch_size)):
            res = 0
            for k in range(batch_size):
                res += h(train_input[k + i * batch_size],theta,theta0)
            res = res / batch_size
            loss.append(((res - train_target[i])[0]) ** 2)
            
            theta0 = theta0 - alpha * (res - train_target[i])
            theta = theta * (1 - alpha * beta) - alpha * (res - train_target[i]) * train_input[i]
    
    
    return theta,theta0,loss

def predict(test_input,theta,theta0):
    for i in range(int(test_input.shape[0])):
        res = h(test_input[i],theta,theta0)
# 评估均方误差
def linear_regression_grade(train_input,train_target,theta,theta0):

    loss = 0
    num = train_input.shape[0]
    cnt = 0


    # tmp_rec1 = []
    # tmp_rec2 = []
    for i in range(int(num)):
        res = h(train_input[i],theta,theta0)
        loss += (((res - train_target[i])[0]) ** 2)

        # tmp_rec1.append(res)
        # tmp_rec2.append(train_target[i][0])
    # plt.plot(tmp_rec1)
    # plt.plot(tmp_rec2)
    # plt.show()
    
    return loss / num


# 梯度下降训练
def tranning(train_input,train_target,theta,theta0):
    theta,theta0,loss = linear_regression(1e-5,0.5,1,train_input,train_target,theta,theta0)
    theta,theta0,tmp = linear_regression(1e-9,0.5,1,train_input,train_target,theta,theta0)
    loss = loss + tmp
    theta,theta0,tmp = linear_regression(1e-9,0.5,2,train_input,train_target,theta,theta0)
    loss = loss + tmp
    # print(linear_regression_grade(train_input,train_target,theta,theta0))

    # theta,theta0,loss = linear_regression(1e-8,0.5,1,train_input,train_target,theta,theta0)
    # print(linear_regression_grade(train_input,train_target,theta,theta0))

    # plt.plot(loss)
    # plt.show()

# 用于调参的辅助函数
def auto_modify(train_input,train_target,test_input,test_target):
    alpha = []
    beta = [1,2,5,10,15,20]
    res = []
    ce = [10,5,3,0.5,0.3,0.1,0.01,0.003,0.001]
    # for a in alpha:
    #     theta,theta0 = init()
    #     theta,theta0,loss = linear_regression(1e-5,0.5,1,train_input,train_target,theta,theta0)
    #     res.append(linear_regression_grade(train_input,train_target,theta,theta0))
    # for i in range(10):
    #     print(alpha[i],res[i])
    # for b in beta:
    #     theta,theta0 = init()
    #     theta,theta0,loss = linear_regression(1e-5,0.5,b,train_input,train_target,theta,theta0)
    #     res.append(linear_regression_grade(train_input,train_target,theta,theta0))
    # for i in range(len(beta)):
    #     print(beta[i],res[i])
    for c in ce:
        theta,theta0 = init()
        theta,theta0,loss = linear_regression(1e-5,c,1,train_input,train_target,theta,theta0)
        res.append(linear_regression_grade(test_input,test_target,theta,theta0))
    for i in range(len(ce)):
        print(ce[i],res[i])

from sklearn import linear_model


# 直接掉库
def use_sklearn(train_input,train_target,test_input,test_target,test_data):
    sqdif = 1000
    while sqdif > 100:
        model = linear_model.LinearRegression()
        model.fit(train_input,train_target)
        res =  model.predict(test_input)
        rt = 0
        for i in range(len(res)):
            rt += (((res[i] - test_target[i])[0]) ** 2)
        sqdif = rt / len(res)

    return sqdif, model.predict(test_data)



train_data,x_train,x_test,y_train,y_test,test_data = get_in_data()

theta,theta0 = init()
tranning(x_train,y_train,theta,theta0)
print(linear_regression_grade(x_test,y_test,theta,theta0))
res = predict(test_data,theta,theta0)

# sqdif, res = use_sklearn(x_train,y_train,x_test,y_test,test_data.data)
# print(sqdif)
resl = np.empty([len(res),2])
for i in range(len(res)):
    resl[i][0] = i
    resl[i][1] = res[i][0]
    
df = pd.DataFrame(resl, columns = ['id','reference'])
df.to_csv('result.csv',sep=',',index = False, header = True)
# theta,theta0 = init()
# theta,theta0,loss = linear_regression(1e-5,0.5,1,x_train,y_train,theta,theta0)
# # theta,theta0,loss = linear_regression(1e-7,0.5,1,x_train,y_train,theta,theta0)
# # theta,theta0,loss = linear_regression(1e-9,0.5,1,x_train,y_train,theta,theta0)

# print(linear_regression_grade(x_test,y_test,theta,theta0))



# auto_modify(x_train,y_train,x_test,y_test)





