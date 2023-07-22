import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import choice
plt.rcParams['font.sans-serif']=['SimSun']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

demand_path = './data/1.demand.csv'
empty_path = './data/3.empty.csv'
pre_path = './data/2.final.csv'  # Z分布
distance_path = './data/5.区域间行驶距离矩阵(m).xlsx'
D = np.loadtxt(demand_path, delimiter=",", dtype='int')[:, 1]  # 出租车需求分布
K = np.loadtxt(empty_path, delimiter=",", dtype='int')[:, 1]  # 出租车空载分布，此处为约束
Z = np.loadtxt(pre_path, delimiter=",", dtype='int')[:, 1]  # 出租车预测分布
df = pd.read_excel(distance_path, header=None)
DIS = np.array(df)  # 距离矩阵

popsize = 100  # 种群数量D.astype(int)
K.astype(int)
Z.astype(int)
#R = 90  # R为网格总数
beta = 0.1  # 变异系数
alpha = 1  # 淘汰系数
# 改变D值，引入Z分布
D = D - Z
D_where = np.where(D < 0)[0]
K[D_where] -= D[D_where]
D[D_where] = 0
'''
delete_where = np.where(K < 30)[0]
K = np.delete(K, delete_where)
D = np.delete(D, delete_where)
Z = np.delete(Z, delete_where)
'''
R = len(K)
'''
DIS = np.delete(DIS, delete_where, axis=0)
DIS = np.delete(DIS, delete_where, axis=1)
'''
function_size = 2  # 两个函数值
generation = 100  # 迭代次数
global iter
fitness_1 = []
fitness_2 = []
print('D',D.sum())
print('K',K.sum())
print('Z',Z.sum())
#  计算pareto的两个函数值， 输入调度矩阵M，区域的出租车需求矩阵D，DIS为网格距离矩阵
# 初始化其中一个个体，K为约束条件，区域i的空载出租车数目
def gen_random(K):
    # 此处变化可能性太小，最好变成每个都单独
    # 先读取每个位置的最大值+1，再考虑其他的
    # M为对角线为0的负对称矩阵，也可不为负对称，因为负对称
    M = np.zeros(shape=(R, R), dtype='int')
    for region_x in range(R):
        region_restrain = K[region_x]
        #pos = np.array(list(range(R)))
        #np.random.shuffle(pos)

        pos = np.argsort(-D)
        zero_pos = np.where(D == 0)[0]
        pos = np.delete(pos, zero_pos)  # 删除为0的位置
        for p in pos:
            if p != region_x and np.sum(M[:, p]) < D[p]:
                region_r = np.random.randint(low = 0, high = region_restrain + 1)
                M[region_x, p] = np.random.randint(low = 0, high = region_r + 1)
                region_restrain -= M[region_x, p]

    for region_x in range(R):
        if M[region_x, :].sum() > K[region_x]:
            print('M检测失败')

    return M

def pareto_fitness(M, D, DIS):
    s = [0.0] * R
    # 列数代表调入数量
    Mi = np.sum(M, axis=0)  # 列求和，求出区域i的调入数量
    #  计算调入区域i的总需求满足度
    for i in range(R):
        if D[i] > Mi[i] > 0:
            s[i] += -Mi[i]/D[i]
        if 0 <= D[i] <= Mi[i]:
            s[i] += -1
    #  计算总共的距离
    distance = [0.0] * R
    for i in range(R):
        distance[i] = sum(M[:, i] * DIS[:, i])

    sum_s = sum(s)/R
    sum_distance = sum(distance)
    # return [sum_s, - sum_distance]  # 都是寻找最大
    return [sum_s, sum_distance], s, distance  # 都是寻找最小

# 交叉选择
def cross_select(pop, D, DIS):
    child_pop = []
    while len(child_pop) < popsize:
        child = np.zeros((R, R), dtype=int)  # 子代
        choice_np = [np.random.randint(low = 0, high = popsize) for _ in range(2)]  # 选择的索引
        gene1 = pop[choice_np[0]]['Gene'].copy()  # M只是为其中一个个体
        gene2 = pop[choice_np[1]]['Gene'].copy()
        _, s1, distance1 = pareto_fitness(gene1, D, DIS)
        _, s2, distance2 = pareto_fitness(gene2, D, DIS)
        for i in range(R):
            if np.less_equal([s2[i], distance2[i]], [s1[i], distance1[i]]).all():  # s2全部比s1小
                child[:, i] = np.copy(gene2[:, i])
            else:
                child[:, i] = np.copy(gene1[:, i])

        # 检查child合法
        for i in range(R):
            region_restrain = K[i]

            if child[i, i] != 0:
                temp = list(range(R))
                temp.pop(i)
                index = choice(temp)
                child[i, index] = child[i, i].copy()
                child[i, i] = 0

            while child[i, :].sum() > region_restrain:  # 如果调出的大于空载分布约束
                    #print(child[i, :].sum(), 'with', region_restrain)
                change_pos = child[i, :].argmax()  # 找到最大值的点的位置
                change_max = child[i, :].max() #获取该行中的最大值
                    #print('change_max', change_max)
                child[i, change_pos] -= np.random.randint(low = 0, high = change_max + 1)
                #print('next', child[i, :].sum(), 'with', region_restrain)
            #print('Done')

        for region_x in range(R):
            if child[region_x, :].sum() > K[region_x]:
                print('cross检测失败')

        #fitness, _, _ = pareto_fitness(child, D, DIS)
        fitness, _, _ = pareto_fitness(child, D, DIS)
        child_pop.append({'Gene': child, 'fitness': fitness, 'pareto_class': None, 'np': None, 'nd': None})

    print('第{}代选择完'.format(iter))
    return child_pop


def mutate(pop):

    for i in range(popsize):  # 循环每一个个体
        for x in range(R):  # 行
            for y in range(R):  # 列
                if np.random.rand() <= beta and x != y:  # 小于变异概率且非对角线
                    pop[i]['Gene'][x, y] = np.random.randint(low = 0, high = K[x] + 1)

            region_restrain = K[x]
            while pop[i]['Gene'][x, :].sum() > region_restrain:  # 如果调出的大于空载分布约束
            # print(child[i, :].sum(), 'with', region_restrain)
                change_pos = pop[i]['Gene'][x, :].argmax()  # 找到最大值的点的位置
                change_max = pop[i]['Gene'][x, :].max()  # 获取该行中的最大值
            # print('change_max', change_max)
                pop[i]['Gene'][x, change_pos] -= np.random.randint(low=0, high=change_max + 1)

        for region_x in range(R):
            if pop[i]['Gene'][region_x, :].sum() > K[region_x]:
                print('mutate检测失败')

    for i in range(popsize):
        #fitness, _, _ = pareto_fitness(pop[i]['Gene'], D, DIS)
        fitness, _, _ = pareto_fitness(pop[i]['Gene'], D, DIS)
        pop[i]['fitness'] = fitness
    print('第{}代交叉完'.format(iter))
    return pop


def crowding_distance(pop, front): #拥挤度排序
    crowding_popsize = len(pop) #排序数量等于上个代码中选择交叉完的种群数量
    crowding = np.zeros(shape=(crowding_popsize,))  # 拥挤距离初始化为0
    for rank in front:  # 遍历每一层Pareto 解 rank为当前等级
        for i in range(function_size):  # 遍历每一层函数值（先遍历群体函数值1，再遍历群体函数值2...）
            value = [pop[A]['fitness'][i] for A in rank]  # 取出rank等级 对应的  目标函数值i 集合
            rank_value = zip(rank, value)  # 将rank,群体函数值i集合在一起
            sort_rank_value = sorted(rank_value, key=lambda x: (x[1], x[0]))  # 先按函数值大小排序，再按序号大小排序

            sort_ranks = [j[0] for j in sort_rank_value]  # 排序后当前等级rank
            sort_values = [j[1] for j in sort_rank_value]  # 排序后当前等级对应的 群体函数值i
            # print(sort_ranki[0],sort_ranki[-1])
            crowding[sort_ranks[0]] = np.inf  # rank 等级 中 的最优解 距离为inf
            crowding[sort_ranks[-1]] = np.inf  # rank 等级 中 的最差解 距离为inf

            # 计算rank等级中，除去最优解、最差解外。其余解的拥挤距离
            for j in range(1, len(rank) - 2):
                if max(sort_values) == min(sort_values):
                    crowding[sort_ranks[j]] = crowding[sort_ranks[j]] + (sort_values[j + 1] - sort_values[j - 1])
                else:
                    crowding[sort_ranks[j]] = crowding[sort_ranks[j]] + (sort_values[j + 1] - sort_values[j - 1]) / (
                        max(sort_values) - min(sort_values))  # 计算距离

    for i in range(len(front)):
        for j in range(len(front[i])):
            pop[front[i][j]]['nd'] = crowding[front[i][j]]

    distance = [[] for i in range(len(front))]  #
    for j in range(len(front)):  # 遍历每一层Pareto 解 rank为当前等级
        for i in range(len(front[j])):  # 遍历给rank 等级中每个解的序号
            distance[j].append(crowding[front[j][i]])

    return pop, distance


def check_dominate(individual1, individual2):
    if np.less(individual1['fitness'], individual2['fitness']).any() \
            and np.less_equal(individual1['fitness'], individual2['fitness']).all():  # 存在一个大于，全部大于或等于
        return True  # 说明pop1优于pop2
    else:
        return False


# 先计算支配数以及支配的索引， 采用快速非支配排序计算类别， 以及计算拥挤度
def Np_get(pop):
    s = [[] for i in range(len(pop))]  # 存放每个个体支配的解的集合
    n = [0 for i in range(len(pop))]  # 每个个体的被支配解的个数
    rank = [float('inf') for i in range(len(pop))]  # 存放每个个体的级别
    front = [[]]  # 存放每个支配的索引
    for p in range(len(pop)):  # 遍历每个个体
        s[p] = []
        n[p] = 0
        for q in range(len(pop)):
            less = 0
            equal = 0
            greater = 0
            for k in range(0, function_size):
                if pop[p]['fitness'][k] > pop[q]['fitness'][k]:  #q小于p
                    less += 1
                if pop[p]['fitness'][k] == pop[q]['fitness'][k]:  #q等于p
                    equal += 1
                if pop[p]['fitness'][k] < pop[q]['fitness'][k]:  # q大于p
                    greater += 1
            if (less + equal == function_size) and (equal != function_size):
                n[p] = n[p] + 1  # q比p,  比p好的个体个数加1
            elif (greater + equal == function_size) and (equal != function_size):
                s[p].append(q)  # q比p差，存放比p差的个体解序号
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
        pop[p]['np'] = n[p]
    #  快速排序
    i = 0  # 第一级
    while front[i]:  # 下一级个体还存在
        Q = []  # 存储下一级个体
        for p in front[i]:
            for q in s[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i += 1
        front.append(Q)

    del front[len(front) - 1]

    pop, crowding = crowding_distance(pop, front)  # 计算拥挤度
    for i in range(len(front)):  # 赋值有问题，front[i]
        for j in front[i]:
            pop[j]['pareto_class'] = i
    return pop, front, crowding

# 精英排序策略
def elitlim(pop, front, crowding):
    popindex = []  # 存储群体编号
    elit_num = alpha * popsize
    for i in range(len(front)):  # 遍历各层
        #print(front[i], crowding[i])
        front_num = len(front[i])
        #print(front_num)
        rank_distance = zip(front[i], crowding[i])  # 当前等级 与当前拥挤距离的集合
        sort_rank_distance = sorted(rank_distance, key=lambda x: (x[1], x[0]), reverse=True)
        # 先按拥挤距离大小排序，再按序号大小排序,逆序
        sort_ranki = [j[0] for j in sort_rank_distance]  # 排序后当前等级rank
        sort_distancei = [j[1] for j in sort_rank_distance]  # 排序后当前等级对应的 拥挤距离i

        if (elit_num - len(popindex)) >= len(sort_ranki):  # 如果X1index还有空间可以存放当前等级i 全部解
                popindex.extend([A for A in sort_ranki])
        elif len(sort_ranki) > (elit_num - len(popindex)):  # 如果X1空间不可以存放当前等级i 全部解
            num = int(elit_num - len(popindex))
            popindex.extend([A for A in sort_ranki[0:num]])


    new_father_pop = [pop[i] for i in popindex]

    return new_father_pop

# 寻找非劣解集
def best_select(pop):
    n = []  # 存放np=0的索引
    best_pop = []
    for i in range(len(pop)):
        if pop[i]['pareto_class'] == 0:
            n.append(i)
            best_pop.append(pop[i])

    return best_pop

def plot_bestpop(best_pop):
    plt.figure()
    best_s = []
    best_distance = []
    for i in range(len(best_pop)):
        s, distance = best_pop[i]['fitness']
        print('distance', distance)
        best_s.append(s)
        # best_distance.append(1/distance)
        best_distance.append(distance)
    plt.scatter(best_distance, best_s )
    plt.xlabel('空载调度距离')
    plt.ylabel('区域需求满足度')
    plt.show()
    # plt.savefig('situation_100_300_K_D_new.png')

def plot_best_sati():
    plt.figure()
    plt.plot(fitness_1, color='red', label='fitness_s')
    plt.legend()
    plt.xlabel('迭代次数')
    plt.title('适应度变化')
    plt.show()
    # plt.savefig('sati_100_300_K_D_new.png')

def plot_best_dis():
    plt.figure()
    plt.plot(fitness_2, color='blue', label='fitness_distance')
    plt.legend()
    plt.xlabel('迭代次数')
    plt.title('适应度变化')
    plt.show()
    # plt.savefig('dis_100_300_K_D_new.jpg')
# 主函数
def NSGA():
    pop = []
    global iter

    for i in range(popsize):  # 初始化
        M = gen_random(K)
        fitness, _, _ = pareto_fitness(M, D, DIS)
        # 初始化类别，个体，适应度，类别, np为支配p的解个数，nd为拥挤度
        pop.append({'Gene': M, 'fitness': fitness, 'pareto_class': None, 'np': None, 'nd': None})
    print('pop初始化个数为', len(pop))
    for iter in range(generation):
        best_s = np.inf
        best_distance = np.inf
        nextpop = cross_select(pop, D, DIS)  #交叉与变异有严重问题
        nextpop = mutate(nextpop)
        pop.extend(nextpop)
        pop, front, crowding = Np_get(pop)  # 计算支配度与拥挤度
        pop = elitlim(pop, front, crowding)
        best_s_index = 0
        for i in range(popsize):
            if pop[i]['fitness'][0] < best_s:
                best_s = pop[i]['fitness'][0]
                best_s_index = i
            if pop[i]['fitness'][1] < best_distance:
                best_distance = pop[i]['fitness'][1]
        print('This {} episode of s is {}'.format(iter, best_s))
        print('This {} episode of distance is {}'.format(iter, best_distance))
        print('s==-1,distance={}'.format(pop[best_s_index]['fitness'][1]))
        fitness_1.append(best_s)
        fitness_2.append(best_distance)

        print('This is {} episode'.format(iter))

    best_pop = best_select(pop)
    #plot_bestpop(best_pop)
    plot_bestpop(pop)
    #plot_bestpop(pop)
    #plot_best_sati()
    #plot_best_dis()
    return pop

pop = NSGA()
