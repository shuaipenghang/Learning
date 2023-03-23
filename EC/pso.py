# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
global weight
def getweight():
    # 惯性权重
    global weight
    weight = 1
    return weight

#返回适应度函数
def func(x):
    # x输入粒子位置
    # y 粒子适应度值
    y = x[0] ** 3 + x[1] ** 3

    return y

#绘制函数图像
def plot_function(x):
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    x1 = x2 = np.arange(start = 0, stop = 4, step = 0.1) #范围为0到4
    X1, X2 = np.meshgrid(x1, x2)
    Y = X1 ** 3 + X2 ** 3
    ax.plot_surface(X1, X2, Y, alpha = 0.9, cstride = 1, rstride = 1, cmap = 'rainbow')
    plt.title('函数图像 最优点为x1=%.2f,x2=%.2f' % (x[0], x[1]))

#输入种群大小，初始化种群，速度，适应度函数
def initpopvfit(sizepop):
    pop = np.zeros((sizepop,2))
    v = np.zeros((sizepop,2))
    fitness = np.zeros(sizepop)

    for i in range(sizepop):
        pop[i] = [(np.random.rand()-0.5)*rangepop[0]*2, (np.random.rand()-0.5)*rangepop[1]*2]
        v[i] = [(np.random.rand()-0.5)*rangepop[0]*2, (np.random.rand()-0.5)*rangepop[1]*2]
        fitness[i] = func(pop[i])
    return pop, v, fitness

def getinitbest(fitness, pop):
    # 群体最优的粒子位置及其适应度值
    gbestpop,gbestfitness = pop[fitness.argmax()].copy(), fitness.max()
    #个体最优的粒子位置及其适应度值,使用copy()使得对pop的改变不影响pbestpop，pbestfitness类似
    pbestpop,pbestfitness = pop.copy(), fitness.copy()

    return gbestpop,gbestfitness,pbestpop,pbestfitness

if __name__ == "__main__":
    w = getweight() #获得权重
    lr = (1.0, 1.0)
    maxgen = 300 #迭代次数
    sizepop = 100
    rangepop = (0.0, 4.0)
    rangespeed = (-1.0, 1.0)

    pop, v, fitness = initpopvfit(sizepop)
    #群体最优，个体最优
    gbestpop, gbestfitness, pbestpop, pbestfitness = getinitbest(fitness, pop)

    result = np.zeros(maxgen)
    for i in range(maxgen):
            t=0.5
            #速度更新
            for j in range(sizepop):
                v[j] += lr[0]*np.random.rand()*(pbestpop[j]-pop[j]) + lr[1]*np.random.rand()*(gbestpop-pop[j])
                #print('每轮速度为', v[j])
            #限定速度
            #v[v<rangespeed[0]] = rangespeed[0]
            #v[v>rangespeed[1]] = rangespeed[1]

            #粒子位置更新
            for j in range(sizepop):
            #pop[j] += 0.5*v[j]
                pop[j] = t*(0.5*v[j])+(1-t)*pop[j]
            #限定位置
            pop[pop<rangepop[0]] = rangepop[0]
            pop[pop>rangepop[1]] = rangepop[1]

            #适应度更新
            for j in range(sizepop):
                fitness[j] = func(pop[j])

            #寻找个体最优与群体最优
            #更新个体的最优值
            for j in range(sizepop):
                if fitness[j] > pbestfitness[j]:
                    pbestfitness[j] = fitness[j]
                    pbestpop[j] = pop[j].copy()

            if pbestfitness.max() > gbestfitness:
                gbestfitness = pbestfitness.max()
                gbestpop = pop[pbestfitness.argmax()].copy()

            result[i] = gbestfitness

    plt.title('适应度函数变化\n最优点为x1=%.2f,x2=%.2f' % (gbestpop[0], gbestpop[1]))
    plt.plot(result)
    plot_function(gbestpop)
    plt.show()