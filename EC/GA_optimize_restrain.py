import numpy as np
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
####################初始化参数#####################
popusize = 50  # 种群数量
generation = 100  # 最大遗传代数
Xmax = 31  # p1上限
Xmin = 0  # p1下限
Ymax = 31  # p2上限
Ymin = 0  # p2 下限
#Load = 50
def calc_f(pops1, pops2):  ##这是生成一个个体的目标函数值
    x1 = pops1 #变量一 p1
    x2 = pops2 #变量二 p2
    y = 0.1*(x1 ** 2) + 4.6*x1 + 0.08*(x2**2) + 80 #计算目标

    return y
print(calc_f(31, 18))
def calc_e(pops1, pops2, LOAD):   ##生成一个个体的惩罚函数值总和
    sumcost = 0#罚项总和

    ee = 0
    """计算约束的惩罚项 LOAD - p1 - p2 <= 0"""
    e1 = LOAD - pops1 - pops2
    ee += max(0, e1) #当不满足这个约束时，e1为正数，由于是求最小值，所以需要加到目标上
    sumcost += ee

    return sumcost #返回罚项

#绘制适应度变化曲线
def plot_fit(best_fit, best_x, LOAD):
    plt.figure()
    iter = list(range(len(best_fit)))
    best_fit = np.reciprocal(best_fit)
    plt.plot(iter, best_fit)
    plt.xlabel('迭代次数')
    plt.ylabel('最优值')
    plt.title('适应度函数变化\n当LOAD为%d时，最优点为x1=%d,x2=%d'%(LOAD, best_x[0], best_x[1]))

#计算适应度函数值与概率，需要加上惩罚项
def sum_fitness(pops1, pops2, LOAD):
    fits_probability = []
    fits = [None] * popusize  # 适应度函数
    fits_penalize = []
    for choose, x1, x2 in zip(range(popusize), pops1, pops2):
        fits[choose] = 1/(calc_f(x1, x2) + calc_e(x1, x2, LOAD)) # 计算适应度函数，计算最小值，故取倒数，取较大的，罚项加入
        fits_penalize.append(calc_e(x1, x2, LOAD))
        if choose == 0:
            fits_probability.append(fits[choose])
        else:
            fits_probability.append(fits[choose] + fits_probability[choose - 1])

    for i in range(popusize):
        fits_probability[i] = fits_probability[i] / fits_probability[-1]  # 概率

    return fits, fits_probability, fits_penalize

#轮盘赌选择法，寻找局部最小
def select_roulette(pops1, pops2, fits_probability):
    child1 = np.zeros(popusize, dtype = int)
    child2 = np.zeros(popusize, dtype = int)

    for i in range(popusize):
        p = np.random.rand() #产生随机概率
        for index in range(len(fits_probability)): #对比概率，寻找选出的染色体
            if index == 0:
                if p < fits_probability[index]:
                    child1[i] = pops1[index].copy()
                    child2[i] = pops2[index].copy()
                    break
            else:
                if p >= fits_probability[index - 1] and p < fits_probability[index]:
                    child1[i] = pops1[index].copy()
                    child2[i] = pops2[index].copy()
                    break

    return child1, child2 #返回两个选择后的子代

def cross(pops, pops_min, pops_max):
    Pc = 0.85  # 交叉率
    """按顺序选择2个个体以概率c进行交叉操作"""
    for i in range(0, pops.shape[0], 2):
        # 产生0-1区间的均匀分布随机数，判断是否需要进行交叉替换
        if np.random.rand() <= Pc:
            mutation_index = np.random.choice(range(len(pops)), size=2)  # 随机选两个交叉，获得索引
            parent1 = np.copy(pops[mutation_index[0]])  # 提取的两个父代
            parent2 = np.copy(pops[mutation_index[1]])
            child1 = (1 - Pc) * parent1 + Pc * parent2  # 这是实数编码 的交叉形式
            child2 = Pc * parent1 + (1 - Pc) * parent2

            # 判断个体是否越限
            if child1 > pops_max or child1 < pops_min:
                child1 = np.random.randint(low = pops_min, high = pops_max + 1)
            if child2 > pops_max or child2 < pops_min:
                child2 = np.random.randint(low = pops_min, high = pops_max + 1)

            pops[mutation_index[0]] = child1
            pops[mutation_index[1]] = child2

    return pops

def mutation(pops, pops_min, pops_max):
    Pm = 0.1  # 变异率
    """变异操作"""
    for i in range(popusize):  # 遍历每一个个体
        # 产生0-1区间的均匀分布随机数，判断是否需要进行变异
        if np.random.rand() <= Pm:
            child = pops_min + Pm * (pops_max - pops_min) #实值变异
            # 判断个体是否越限，越界即再生成一个随机数
            if child > pops_max or child < pops_min:
                child = np.random.randint(low = pops_min, high = pops_max + 1)

            pops[i] = child

    return pops

# 子代与父代一对一竞争，保留优秀的个体，输入适应度函数，罚项
def update_best(pops1, pops2, parent_fitness, parent_e, child1, child2, child_fitness, child_e):
    """
        子代与父代一对一竞争，保留优秀的个体
        :param pops1, pops2: 父辈个体
        :param parent_fitness:父辈适应度值
        :param parent_e    ：父辈惩罚项
        :param child1, child2:  子代个体
        :param child_fitness 子代适应度值
        :param child_e  ：子代惩罚项

        :return: 父辈 和子代中较优者、惩罚项

        """
    # 规则1，如果 parent 和 child 都没有违反约束，则取适应度大的
    if parent_e <= 0.0000001 and child_e <= 0.0000001:
        if parent_fitness >= child_fitness:
            return pops1, pops2, parent_fitness
        else:
            return child1, child2, child_fitness
    # 规则2，如果child违反约束而parent没有违反约束，则取parent
    if parent_e < 0.0000001 and child_e >= 0.0000001:
        return pops1, pops2, parent_fitness
    # 规则3，如果parent违反约束而child没有违反约束，则取child
    if parent_e >= 0.0000001 and child_e < 0.0000001:
        return child1, child2, child_fitness
    # 规则4，如果两个都违反约束，则取适应度值大的
    if parent_fitness >= child_fitness:
        return pops1, pops2, parent_fitness
    else:
        return child1, child2, child_fitness

#GA算法，输入两个变量，LOAD是变化的值
def GA(pops1, pops2, LOAD):
    best_fit_list = [] #存放最优函数
    best_x_list = [] #存放最优值
    for iter in range(generation): #迭代次数
        fits, fits_probability, fits_penalize = sum_fitness(pops1, pops2, LOAD)  #父代结果
        child1, child2 = select_roulette(pops1, pops2, fits_probability)  #轮盘赌
        child1 = cross(child1, Xmin, Xmax) #交叉
        child2 = cross(child2, Ymin, Ymax)
        child1 = mutation(child1, Xmin, Xmax) #变异
        child2 = mutation(child2, Ymin, Ymax)

        child_fits, fits_probability, child_fits_penalize = sum_fitness(child1, child2, LOAD) #更新后的适应度函数

        for i in range(popusize):  # 一对一竞争，选出父代与子代中最优的
            pops1[i], pops2[i], fits[i] = \
            update_best(pops1[i], pops2[i], fits[i], fits_penalize[i], child1[i], child2[i], child_fits[i], child_fits_penalize[i])


        best_fit = np.max(fits) #寻找适应度函数最大
        best_fit_index = np.argmax(fits) #最大的索引
        best_fit_list.append(best_fit)
        best_x_list.append([pops1[best_fit_index], pops2[best_fit_index]])

    best_fit = best_fit_list[-1]
    best_x = best_x_list[-1]
    plot_fit(best_fit_list, best_x, LOAD) #绘制适应度变化图

    return 1/best_fit, best_x


if __name__ == '__main__':
    for LOAD in (45, 50):
        pops1 = np.random.randint(low = Xmin, high = Xmax + 1, size = popusize)  #第一个变量，限定为整数
        pops2 = np.random.randint(low = Ymin, high = Ymax + 1, size = popusize)  #第二个变量
        best_fit, best_x = GA(pops1, pops2, LOAD) #GA运行
        print("最优值是：%.5f" % best_fit)
        print("最优解是：xy=" , best_x)
        print("LOAD = %d" %LOAD)

    plt.show()

