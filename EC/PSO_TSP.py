import random
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

num = 10  # 15个城市
popusize = 100  # 个体数
generation = 100  # 迭代次数


# 计算距离
def DistEur(a, b):  # 输入对应列表
    sum1 = 0
    for i in range(len(a)):
        val = np.power(a[i] - b[i], 2)
        sum1 = sum1 + val
    dist = np.sqrt(sum1)

    return dist


def GetDistance(DataSet):
    distance_matrix = np.zeros((num, num))  # 距离矩阵
    for a in range(num):
        for b in range(num):
            distance_matrix[a, b] = DistEur(DataSet[a, :], DataSet[b, :])

    return distance_matrix


def CreateDataSet(num):
    np.random.seed(4)
    DataSet = np.random.uniform(0, 100.0, [num, 2])  # 索引即是城市位置
    return DataSet


def PlotMap(DataSet):
    plt.figure()
    x = DataSet[:, 0]  # x轴
    y = DataSet[:, 1]  # y轴
    plt.title('Map')
    plt.plot(x, y, 'ro')


# 绘制访问顺序
def PlotVisit(DataSet, visit):
    plt.figure()
    x = DataSet[:, 0]  # x轴
    y = DataSet[:, 1]  # y轴
    for i in range(num):
        if i < num - 1:
            plt.plot([x[visit[i]], x[visit[i + 1]]], [y[visit[i]], y[visit[i + 1]]], 'ro-')
        else:
            plt.plot([x[visit[i]], x[visit[0]]], [y[visit[i]], y[visit[0]]], 'ro-')


def SumDistance(ReGreCode, distance_matrix):  # 输入解码后，计算总距离
    SumDist = 0
    for i in range(num):
        if i < num - 1:
            SumDist = SumDist + distance_matrix[ReGreCode[i], ReGreCode[i + 1]]
        else:  # 首尾相接
            SumDist = SumDist + distance_matrix[ReGreCode[i], ReGreCode[0]]

    return 1 / SumDist  #适应度函数为倒数
    #return SumDist

def Exchange(A, B): #A-B
    SO = []
    for i in range(len(A)):
        location = np.where(B == A[i])[0][0]
        if location != i:
            SO.append([i, location])
            temp = B[i]
            B[i] = np.copy(B[location])
            B[location] = temp

    return np.array(SO)

def Fiel(velocity, r): #保留，输入速度与学习因子
    index = len(velocity)
    newvlen = np.round(index * r)
    newv = velocity[: int(newvlen-1)]

    return newv

def UpdateVelocity(velocity, visit, weight, pops, best_pops):
    r1 = 0.7
    r2 = 0.7
    popsx, popsy = np.shape(visit)
    VelocityPops = np.array([None] * popsx)
    VelocityBestPops = np.array([None] * popsx)
    NewVelocity = np.array([None] * popsx)

    for i in range(popsx): #每个个体的分速度
        VelocityPops[i] = Exchange(pops[i], visit[i])
        VelocityBestPops[i] = Exchange(best_pops, visit[i])
        VelocityPops[i] = Fiel(VelocityPops[i], r1)
        VelocityBestPops[i] = Fiel(VelocityBestPops[i], r2)
        velocity[i] = Fiel(velocity[i], weight)

    for i in range(popsx):
        NewVelocity[i] = velocity[i]
        NewVelocity[i] = np.append(NewVelocity[i], VelocityPops[i])
        NewVelocity[i] = np.append(NewVelocity[i], VelocityBestPops[i])
        NewVelocity[i] = NewVelocity[i].reshape(-1, 2)

    return NewVelocity


def initVelocity(visit):
    popsx, popsy = np.shape(visit)
    velocity = np.array([None] * popsx)
    for k in range(len(visit)):
        index = np.random.choice(range(num), size=2)
        velocity[k] = visit[k][index]

    return velocity

def UpdatePops(visit, velocity): #改变位置
    for i in range(len(visit)):
        for k in velocity[i]:
            temp = np.copy(visit[i][int(k[0])])
            visit[i][int(k[0])] = np.copy(visit[i][int(k[1])])
            visit[i][int(k[1])] = temp

    return visit


def Selection(visit, fits, distance_matrix):
    pops = np.copy(visit)  # 每个子代的最优解对应的位置
    new_fits = np.array([0.0] * len(visit)) #单个子代的最优适应度
    weight = 0.5
    best_pops = np.array([None] * num)
    best_fit_list = np.array([])
    for iter in range(generation): #迭代次数

        best_fit = np.max(fits) #fits存储目标，种群最优适应度
        best_fit_index = np.argmax(fits)
        best_pops = visit[best_fit_index]  # 种群最优适应度对应的位置

        best_fit_list = np.append(best_fit_list, 1/best_fit)

        for k in range(len(visit)):#更新pops
            new_fits[k] = SumDistance(pops[k], distance_matrix)
            fits[k] = SumDistance(visit[k], distance_matrix)
            if new_fits[k] < fits[k]: #更新
                pops[k] = np.copy(visit[k])
                new_fits[k] = np.copy(fits[k])

        velocity = initVelocity(visit) #初始化速度
        velocity = UpdateVelocity(velocity, visit, weight, pops, best_pops) #改变速度
        visit = UpdatePops(visit, velocity) #更新位置

    best_fit_index = np.argmax(fits)
    best_pops = visit[best_fit_index]  # 种群最优适应度对应的位置
    PlotVisit(DataSet, best_pops)
    PlotFits(best_fit_list)
    return visit, fits

def PlotFits(fits):
    plt.figure()
    iter = list(range(len(fits)))
    plt.plot(iter, fits)
    plt.xlabel('迭代次数')
    plt.ylabel('最短距离')
    plt.title('适应度函数变化')


if __name__ == '__main__':

    fits = [None] * popusize  # 适应度函数
    DataSet = CreateDataSet(num)  # 创建城市
    np.random.seed(1)  # 恢复完全随机的种子
    visit = np.array([random.sample([i for i in list(range(num))], num) for j in range(popusize)])  # 访问顺序
    distance_matrix = GetDistance(DataSet)  # 获得各个城市间的距离
    for i in range(popusize):
        fits[i] = SumDistance(visit[i], distance_matrix)  # 初代的适应度
    best_fit = np.max(fits)
    best_fit_index = np.argmax(fits)
    print('初代的最优值为: %f' % (1/best_fit))
    PlotVisit(DataSet, visit[best_fit_index])
    plt.title('初代最优轨迹')

    visit, fits = Selection(visit, fits, distance_matrix)

    plt.show()