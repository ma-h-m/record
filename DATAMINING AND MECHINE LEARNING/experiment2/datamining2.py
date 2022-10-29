import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_in_data(): # 读入数据，构建数据集、正则化并拆分

    df = pd.read_csv('x_train.csv')
    df2 = pd.read_csv('task1_label_train.csv')
    df3 = pd.read_csv('task2_label_train.csv')
    return df,df2,df3
# get_in_data()

def get_in_data_(): # 读入数据，构建数据集、正则化并拆分

    df = pd.read_csv('x_train_test.csv')
    df = df.iloc[:,1:]

    df2 = pd.read_csv('task1_label_train_test.csv')
    df2 = df2.iloc[:,1:]
    df3 = 0
    return df,df2,df3

features,labels1,labels2 = get_in_data()





def get_entropy(plus,minus):
    
    if plus == 0 or minus == 0:
        return 0
    sum = plus + minus
    return - plus / sum * math.log(plus/sum) - minus / sum * math.log(minus/sum)

def get_entropy_of_node(labels):
    l = len(labels)
    cnt1 = sum(labels)
    cnt2 = l - cnt1
    rate1 = cnt1 / (l)
    rate2 = cnt2 / l
    return -rate1 * math.log(rate1) - rate2 * math.log(rate2)

# 获取选择一个特征后的分割位点于相应的熵
# 以空间换时间，每次划分后不重新计数，而是在排序完成后，将lable也一同排序，然后开始累计。而非每次重新排序

# 记得每次调用此函数前一定要deepcopy一个副本作为参数传递过来
def get_entropy_and_division_of_column(features,labels):
    
    # 完成重新排序
    lens = len(labels)
    rank = np.argsort(features)
    features = features[rank] 
    labels = labels[rank]

    # 寻找并统计各个不同分割位点
    # 左右节点统计量的初始化

    mini_s2_sum = 1e9
    division = -1

    for i in range(lens - 1):
        tmp = labels[:(i + 1)].var() * (i + 1) + labels[(i + 1):].var() * (lens - i - 1)
        if tmp < mini_s2_sum:
            division = (features[i] + features[i + 1]) / 2
            mini_s2_sum = tmp
    
    return mini_s2_sum, division



    
class tree_node(object):
    def __init__(self) -> None:
        self.l = []
        self.r = []
        self.division = -1
        self.feature = -1
        # 这个feature是该节点的特征编号
        self.val = -1
        # 使用该点平均值作为val

    def set_node(self,l,r,feature,division,val):
        self.l = l
        self.r = r
        self.division = division
        self.feature = feature
        self.val = val
    def predict(self,features):
        if self.l == None:
            return self.val
        if features[self.feature] > self.division:
            return self.r.predict(features)
        return self.l.predict(features)

# 决策树建树

def create_tree(features,labels,depth):
    if depth == 0:
        return None
    if len(features) == 0:
        return None
    features_num = len(features[0])
    mini_entropy = 1e9
    division = -1
    feature_rec = -1
    for i in range(features_num):
        feature = copy.deepcopy(features[:,i])
        label = copy.deepcopy(labels)
        tmp_entropy, tmp_division = get_entropy_and_division_of_column(feature,label)
        if tmp_entropy < mini_entropy:
            mini_entropy = tmp_entropy
            division = tmp_division 
            feature_rec = i
    
    node = tree_node()
    node.set_node([],[],feature_rec,division,sum(labels) / len(labels))
    node.l = create_tree(features[features[:,feature_rec] <= division],labels[features[:,feature_rec] <= division],depth-1)
    node.r = create_tree(features[features[:,feature_rec] > division],labels[features[:,feature_rec] > division],depth-1)

    return node

def test(features,labels,trees):
    for i in range(len(features)):
        print(trees.predict(features[i]),'   ',labels[i])


class GBDT(object):
    def __init__(self,num,depth):
        self.trees = []
        self.num = num # 树的数量
        self.depth = depth
    def predict(self,features):
        rt = 0
        if len(self.trees) == 0:
            return 0
        for i in self.trees:
            rt += i.predict(features)
        return rt

    def calcuate_grad(self,features):
        return [self.predict(x) for x in features]
    def train(self,features,labels):
        la = copy.deepcopy(labels)

        for i in range(self.num):
            # 计算残差（这里默认使用二阶矩作为损失函数）
            la = labels- self.calcuate_grad(features)
            # print(self.calcuate_grad(features))
            # print(la)
            self.trees.append(create_tree(np.array(features),np.array(la),self.depth))
        
        


print(features)
print(labels1)

testObject = GBDT(5,5)
testObject.train(np.array(features),np.array(labels1)[:,0])

test(np.array(features),np.array(labels1)[:,0],testObject)
