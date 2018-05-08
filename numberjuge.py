# coding: utf-8
from numpy import *
import operator
from os import listdir


'''
识别手写数字
'''

def imgetovector(filename):
    '''

    :param filename: 文件名
    :return: 将图像32*32的数据转化为1*1024的一维矩阵
    '''
    res = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        str = fr.readline()
        for j in range(32):
            res[0, 32 * i + j] = int(str[j])
    return res


def classfyKnn(inx, dataSet, lables, k):
    '''

    :param inx: 待预测的特征值
    :param dataSet: 训练样本特征值
    :param lables: 训练样本lable值
    :param k: 参数k
    :return: 预测的lable值
    '''
    # 训练样本的行数
    lines = dataSet.shape[0]
    # 做差求距离
    diffMat = tile(inx, (lines, 1)) - dataSet
    diffMat = diffMat**2
    # 对每一行的距离求和转化为一维矩阵
    distance = diffMat.sum(axis=1)
    distance = distance**0.5
    # 升序
    sortedIndex = distance.argsort()
    # 取全k个,统计lable值，返回总数最多的lable
    listk = {}
    for i in range(k):
        key = lables[sortedIndex[i]]
        listk[key] = listk.get(key, 0) + 1
    res = sorted(listk.iteritems(), key=operator.itemgetter(1), reverse=True)
    #放回最大值对应的lable值
    return res[0][0]


def TestNumber():
    #1.先处理样本数据
    trainingFileList = listdir("/home/wendong/PycharmProjects/MachineLearning/input/2.KNN/trainingDigits")
    lines = len(trainingFileList)
    dataSet = zeros((lines,1024))
    lables = []
    for i in range(lines):
        fileName = trainingFileList[i]
        str = fileName.split(".")[0]
        lables.append( int(str.split("_")[0]))
        dataSet[i:] = imgetovector('/home/wendong/PycharmProjects/MachineLearning/input/2.KNN/trainingDigits/%s' % fileName)

    #测试数据
    testFiledList = listdir("/home/wendong/PycharmProjects/MachineLearning/input/2.KNN/testDigits")
    testLines = len(testFiledList)
    print('样本数据：',dataSet)
    print('目标值:',lables)
    errcount = 0
    for i in range(testLines):
        pre = int((testFiledList[i].split(".")[0]).split("_")[0])
        inx = imgetovector('/home/wendong/PycharmProjects/MachineLearning/input/2.KNN/testDigits/%s' % testFiledList[i])
        res = classfyKnn(inx,dataSet,lables,5)
        print('数字%s被预测成%s' % (pre,res))
        if pre !=res:
            errcount = errcount + 1
    print('错误率:',errcount/float(lines))

if __name__ == '__main__':
    TestNumber()
