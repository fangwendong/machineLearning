# coding: utf-8
# from sklearn import tree
from numpy import *
# import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import operator


# txt文件读取
def file2matrix(filename):
    """
    Desc:
        导入训练数据
    parameters:
        filename: 数据文件路径
    return:
        数据矩阵 returnMat 和对应的类别 classLabelVector
    """
    fr = open(filename)
    # 获得文件中的数据行的行数
    numberOfLines = len(fr.readlines())
    # 生成对应的空矩阵
    # 例如：zeros(2，3)就是生成一个 2*3的矩阵，各个位置上全是 0
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # str.strip([chars]) --返回已移除字符串头尾指定字符所生成的新字符串
        line = line.strip()
        # 以 '\t' 切割字符串
        listFromLine = line.split('\t')
        # 每列的属性数据
        returnMat[index, :] = listFromLine[0:3]
        # 每列的类别数据，就是 label 标签数据
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # 返回数据矩阵returnMat和对应的类别classLabelVector
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    Desc:
        归一化特征值，消除特征之间量级不同导致的影响
    parameter:
        dataSet: 数据集
    return:
        归一化后的数据集 normDataSet. ranges和minVals即最小值与范围，并没有用到

    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """
    # 计算每种属性的最大值、最小值、范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 极差
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 将最小值之差除以范围组成矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals


def classify0(inX, dataSet, labels, k):
    print('dataSet:', dataSet)
    dataSetSize = dataSet.shape[0]
    # 距离度量 度量公式为欧氏距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    # 将距离排序：从小到大
    sortedDistIndicies = distances.argsort()
    print('sortedDistIndicies:', sortedDistIndicies)
    # 选取前K个最短距离， 选取这K个中最多的分类类别
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #     字典排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games ?"))
    ffMiles = float(raw_input("frequent filer miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('test.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print "You will probably like this person: ", resultList[classifierResult - 1]


if __name__ == '__main__':
    classifyPerson()
    # resultList = ['not at all', 'in small doses', 'in large doses']
    # percentTats = float(raw_input("percentage of time spent playing video games ?"))
    # ffMiles = float(raw_input("frequent filer miles earned per year?"))
    # iceCream = float(raw_input("liters of ice cream consumed per year?"))
    # datingDataMat, datingLabels = file2matrix('test.txt')
    # normMat, ranges, minVals = autoNorm(datingDataMat)
    # inArr = array([ffMiles, percentTats, iceCream])
    # classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    # print "You will probably like this person: ", resultList[classifierResult - 1]
    # mat, vector = file2matrix('/home/wendong/PycharmProjects/sklearn/test.txt')
    # print(mat)
    # print(vector)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # normalMat = autoNorm(mat)
    # ax.scatter(normalMat[:, 0], normalMat[:, 1], 15.0 * array(vector), 15.0 * array(vector))
    # plt.show()

'''
KNN 
1.在数据样本中找到k个最接近目标距离的样本s1
2.在s1中统计出现最多次数的特征值，得出结论
'''