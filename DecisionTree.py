# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 16:42:18 2018

@author: huangjs
"""
from math import log
##
 # filePath: 待读取的文件位置
 # return dataSet：样本数据，featuresTitle：特征列表
##
def loadData(filePath):
    dataSet = []
    featuresTitle = []
    
    ## 读取文件
    fr = open(filePath)
    # 读取特征名
    featuresTitle = fr.readline().strip().split()[0 : 4]
    # 读取样本数据
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataSet.append(lineArr)
    return dataSet, featuresTitle

##
 # return 传入数据集的信息熵
##
def calEntropy(dataSet):
    
    ## 统计各个类别出现的次数
    labelCount = {}
    for sample in dataSet:
        label = sample[-1]
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1
    
    ## 计算信息熵
    entropy = 0.0
    for key in labelCount:
        prob = labelCount[key] / len(dataSet)
        entropy += -prob*log(prob, 2)
    return entropy  

##
 # 过滤数据，依据数据集（dataSet）某一特征（featureIndex）的取值（value）过滤数据
 # return 过滤后的数据集，共两列，第一列为该特征的所有取值，第二列为对应的类别
##
def filterDataSet(dataSet, featureIndex, featureValue):
    
    fDataSet = []
    for sample in dataSet:
        if sample[featureIndex] == featureValue:
            fDataSet.append([sample[featureIndex], sample[-1]])
    return fDataSet
        
##
 # dataSet: 整个样本数据集
 # totalEntropy：整个样本的信息熵
 # featureIndex：特征所在列，0-base system
 # return 该特征的信息增益
## 
def calFeatureInfoGain(dataSet, totalEntropy, featureIndex):
    #样本数
    n = len(dataSet)
    # 筛选出该列数据
    featureDataSet = [sample[featureIndex] for sample in dataSet]
    # 获取该特征的对应的子树
    featureValues = set(featureDataSet)
    
    ## 1. 计算各子树的信息熵
    subtreeEntropy = {}
    for value in featureValues:
        fDataSet = filterDataSet(dataSet, featureIndex, value)
        subtreeEntropy[value] = calEntropy(fDataSet)
    
    ## 2. 计算该特征的信息熵   
    # 统计特征取值对应的样本数(特征取值=>样本数)
    featureValuesCount = {}
    for featureVal in featureDataSet:
        if featureVal not in featureValuesCount.keys():
            featureValuesCount[featureVal] = 0
        featureValuesCount[featureVal] += 1
    # 计算特征信息熵
    featureEntropy = 0.0
    for value in featureValues:
        featureEntropy += (float(featureValuesCount[value]) / n) * subtreeEntropy[value]
    
    ## 3. 计算特征的信息增益
    featureInfoGain = totalEntropy - featureEntropy
    return featureInfoGain

def chooseBestFeature(dataSet, totalEntropy):
    matchedIndex = -1
    maxInforGain = 0.0
    for featureIndex in range(len(dataSet[0]) - 1):
        currentInfoGain = calFeatureInfoGain(dataSet, totalEntropy, featureIndex)
        if(currentInfoGain > maxInforGain):
            maxInforGain = currentInfoGain
            matchedIndex = featureIndex
    return matchedIndex
##
 # dataSet: 样本数据集
  # featuresTitle：特征列表
 # featureIndex：特征所在列，0-base system
 # return 去除包含某一特征取值所在行的数据
## 
def splitDataSet(dataSet, featuresTitle, featureIndex, featureValue):
   
    ## 分割数据集
    spDataSet = []
    for sample in dataSet:
        if sample[featureIndex] == featureValue:
            tmp = sample[0 : featureIndex]
            tmp.extend(sample[featureIndex + 1 :])
            spDataSet.append(tmp)
    
    ## 分割特征
    spFeaturesTitle = featuresTitle[0 : featureIndex]
    spFeaturesTitle.extend(featuresTitle[featureIndex + 1 :])
    return spDataSet, spFeaturesTitle

def decisionTree(dataSet, featuresTitle, totalEntropy):
    tree = {}
    
    # 获取样本类别
    classList = [sample[-1] for sample in dataSet]
    
    ## 终止条件：所有的样本全部属于一类
    if classList.count(classList[0]) == len(classList):
        return classList[0] + "**"
    
    ## 终止条件：遍历完所有的特征
    if len(dataSet) == 1:
        return classList[0]
    
    ## 递归
    # 获取该分裂节点的属性
    bestFeatureIndex = chooseBestFeature(dataSet, totalEntropy)
    bestFeature = featuresTitle[bestFeatureIndex]
    # 分割数据
    featureDataSet = [sample[bestFeatureIndex] for sample in dataSet]
    tree = {bestFeature: {}}
    subtrees = set(featureDataSet)
    for subtree in subtrees:
        spDataSet, spFeaturesTitle = splitDataSet(dataSet, featuresTitle, bestFeatureIndex, subtree)
        tree[bestFeature][subtree] = decisionTree(spDataSet, spFeaturesTitle, totalEntropy)
    return tree
################# function call ##################
dataSet, featuresTitle = loadData("./support/tennis.txt")
totalEntropy = calEntropy(dataSet)
#outlookInfoGain = calFeatureInfoGain(dataSet, totalEntropy, 0)
#bestFeatureIndex = chooseBestFeature(dataSet, totalEntropy)
#bestFeature = featuresTitle[bestFeatureIndex]
tree = decisionTree(dataSet, featuresTitle, totalEntropy)
    

