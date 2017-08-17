# coding=utf-8

'''
Date: August 14 2017
Author：Chenhao
Description: simplified SMO Algorithm
'''

from numpy import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#读取文档内数据，数据格式为三列，前两列包含二维数据，第三列为label
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#用于在区间范围内选择一个整数
#输入函数i代表第一个alpha的下标，m是所有alpha的数目
def selectJrand(i,m):
    j=i             #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

#用于在数值太大的情况下进行调整
#用于调整大于H小于L的alpha值
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

#简化版的SMO算法
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):     #五个输入参数对应表示数据集，类别标签，常数C，容错率，退出前最大循环次数
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()    #将输入数据转化为matrix的形式，将类别标签进行转置
    b = 0; m,n = shape(dataMatrix)    #将常数b进行初始化，m，n分别表示数据集的行和列
    alphas = mat(zeros((m,1)))      #将alpha初始化为m行1列的全零矩阵
    iter = 0        #该变量用于存储是在没有任何alpha改变的情况下遍历数据集的次数
    while (iter < maxIter):     #当遍历数据集的次数超过设定的最大次数后退出循环
        alphaPairsChanged = 0       #该参数用于记录alpha是否已进行优化，每次循环先将其设为0
        for i in range(m):      #将数据集中的每一行进行测试
            #multiply是numpy中的运算形式，将alpha与标签相乘的结果的转置与对应行的数据相乘再加b
            #fxi表示将该点带入后进行计算得到的预测的类别，若Ei=0，则该点就在回归线上
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            #label[i]*Ei代表误差，如果误差很大可以根据对应的alpha值进行优化，同时检查alpha值使其不能等于0或C
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)       #随机选取另外一个数作为j，也就是选取另外一行数据
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])       #采用与上述相同的方式来计算j行的预测误差
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();     #暂存变量，便于后面对其进行比对
                #调整H与L的数值，便于用来控制alpha的范围
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue     #continue在python中代表本次循环结束直接运行下一次循环
                #eta代表的是alphas[j]的最优修改量，如果eta大于等于0则需要退出for循环的当前迭代过程
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue
                #给alphas[j]赋新值，同时控制alphas[j]的范围在H与L之间
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                #然后检查alphas[j]是否有轻微改变，如改变很小，就进行下一次循环
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                #将alpha[i]与alpha[j]同时进行改变，改变的数值保持一致，但是改变的方向相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                #在完成alpha[i]与alpha[j]的优化之后，给这两个alpha设置一个常数项b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1      #该参数用于记录alpha是否已进行优化
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)      #iter表示遍历的次数
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas

def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def plot():
    dataplot = array(dataArr)
    n = shape(dataplot)[0]
    plotx1 = []; ploty1 = []
    plotx2 = []; ploty2 = []
    for i in range(n):
        if int(labelArr[i]) == 1:
            plotx1.append(dataplot[i,0])
            ploty1.append(dataplot[i,1])
        else:
            plotx2.append(dataplot[i, 0])
            ploty2.append(dataplot[i, 1])
    fig = plt.figure()
    ax =fig.add_subplot(111)
    ax.scatter(plotx1,ploty1,c='red')
    ax.scatter(plotx2,ploty2,c='blue')
    plotx = arange(0.0, 10.0, 0.1)
    ploty = -(ws_float0 * plotx + b_float) / ws_float1
    ax.plot(plotx, ploty)
    plt.show()


dataArr,labelArr = loadDataSet('testSet.txt')
b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)
ws = calcWs(alphas,dataArr,labelArr)
print ws
b_float = float(b[0])
ws_float0 = float(ws[0])
ws_float1 = float(ws[1])
'''
for i in range(100):
    if alphas[i]>0.0:
        print dataArr[i],labelArr[i]
'''
plot()