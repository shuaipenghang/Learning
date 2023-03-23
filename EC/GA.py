import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

#创建城市，绘制城市
def create_map():
    #20个城市坐标
    map = np.array([[88, 16], [42, 76], [5, 76], [69, 13],
                    [73, 56], [100, 100], [22, 92], [48, 74],
                    [73, 46], [39, 1], [51, 75], [92, 2],
                    [101, 44], [55, 26], [71, 27], [42, 81],
                    [51, 91], [89, 54], [33, 18], [40, 78]])

    plt.figure()
    map_x = map[:, 0] #x轴
    map_y = map[:, 1] #y轴
    plt.title('Map of Cities')
    plt.plot(map_x, map_y, 'ro')

    return map

#绘制适应度函数变化曲线
def PlotFits(fits):
    plt.figure()
    iter = list(range(len(fits)))
    plt.plot(iter, fits)
    plt.xlabel('迭代次数')
    plt.ylabel('最短距离')
    plt.title('适应度函数变化')

#绘制城市访问路径
def PlotVisit(map, visit, best_fit):
    plt.figure()
    map_x = map[:, 0]
    map_y = map[:, 1]
    num = len(map)
    plt.title('最短轨迹，最短为：%.2f m' % best_fit)
    for i in range(num):
        if i < num - 1:
            plt.plot([map_x[visit[i]], map_x[visit[i + 1]]], [map_y[visit[i]], map_y[visit[i + 1]]], 'ro-')
        else:
            plt.plot([map_x[visit[i]], map_x[visit[0]]], [map_y[visit[i]], map_y[visit[0]]], 'ro-') #尾部相连

#计算两点的欧式距离
def distance_eur(a, b): #输入对应列表
    sum1 = 0
    for i in range(len(a)):
        val = np.power(a[i]-b[i], 2)
        sum1 = sum1 + val
    dist = np.sqrt(sum1)

    return dist

#获得距离矩阵，每个城市相互的距离
def get_distance(map):
    num = len(map)
    distance_matrix = np.zeros((num, num))  # 距离矩阵
    for a in range(num):
        for b in range(num):
            distance_matrix[a, b] = distance_eur(map[a, :], map[b, :])

    return distance_matrix

#计算该路径下所花费的距离
def sum_distance(visit, distance_matrix):
    sum_dist = 0
    city_num = len(visit)
    for i in range(city_num):
        if i < city_num - 1:
            sum_dist = sum_dist + distance_matrix[visit[i], visit[i + 1]]
        else:
            sum_dist = sum_dist + distance_matrix[visit[i], visit[0]]

    return round(sum_dist, 1) #四舍五入到一位

#基因交叉操作
def Cross(pops_1, pops_2, distance_matrix):
    PCross = 0.95  # 交叉概率
    popusize = len(pops_1)
    num = len(pops_1[0])
    new_pops = np.array([], dtype=int)
    new_fits = np.array([], dtype=int)
    for i in range(popusize):
        P = np.random.rand()  # 交换的概率
        if P <= PCross:  # 发生变异
            start = np.random.randint(1, num - 2)  # 随机的交换区间
            end = np.random.randint(1, num - 2)
            if start > end:
                temp = start
                start = end
                end = temp
            mutation_index = np.random.choice(range(len(pops_1)), size=2)  # 随机选两个交叉，获得索引
            father1 = np.copy(pops_1[mutation_index[0]])  # 提取的两个父代容易变得同一
            father2 = np.copy(pops_2[mutation_index[1]])
            DNA_f1 = np.copy(father1[start:end + 1])
            DNA_f2 = np.copy(father2[start:end + 1])
            Con_DNA = []  # 相同的染色体
            for x in range(len(DNA_f1)):
                for y in range(len(DNA_f2)):
                    if DNA_f1[x] == DNA_f2[y]:
                        Con_DNA.append(DNA_f2[y])
            # 标记重复
            for k in DNA_f1:
                father2[np.where(father2 == k)[0]] = -1
            for k in DNA_f2:
                father1[np.where(father1 == k)[0]] = -1
            # 交换子代
            father2[start:end + 1] = np.copy(DNA_f1)
            father1[start:end + 1] = np.copy(DNA_f2)
            # 删除相同的作为后面的补充
            for z in Con_DNA:
                DNA_f1 = np.delete(DNA_f1, np.where(DNA_f1 == z)[0])
                DNA_f2 = np.delete(DNA_f2, np.where(DNA_f2 == z)[0])
            father1_sign = np.where(father1 == -1)[0]
            father2_sign = np.where(father2 == -1)[0]
            for d in range(len(DNA_f1)):
                father1[father1_sign[d]] = DNA_f1[d]
                father2[father2_sign[d]] = DNA_f2[d]
            new_pops = np.append(new_pops, father1).reshape(-1, num)
            new_fits = np.append(new_fits, sum_distance(father1, distance_matrix))
        else: #没有发生交叉的情况使用其中一个基因作为子代
            new_pops = np.append(new_pops, pops_1[i]).reshape(-1, num)
            new_fits = np.append(new_fits, sum_distance(pops_1[i], distance_matrix))

    return new_pops, new_fits

#变异，随机选个变异，进行交叉
def Mutation(new_pops, new_fits, distance_matrix):
    PMutation = 0.20 #变异概率
    popusize = len(new_pops)
    num = len(new_pops[0])
    for i in range(popusize):
        P = np.random.rand()
        if P <= PMutation:
            Mutation1 = np.random.randint(1, num - 2)  # 随机变异的点，多个点变异
            Mutation2 = np.random.randint(1, num - 2)  # 随机变异的点，多个点变异
            if Mutation1 != Mutation2:
                change = np.copy(new_pops[i][Mutation1]) #随机变异后的数
                new_pops[i][Mutation1] = new_pops[i][Mutation2] #交换变异后的基因
                new_pops[i][Mutation2] = change
                new_fits[i] = sum_distance(new_pops[i], distance_matrix)

    return new_pops, new_fits

#选择操作，本次采用锦标赛选择法，每次选择5个进行选择的竞争
def tournament_select(visit, fits):
    city_num = len(visit[0])
    new_pops = np.zeros((len(visit), city_num), dtype=int)  # 新的子代
    new_fits = np.array([0.0] * len(visit))
    for i in range(len(visit)): #创造相同数量的子代
        tourna_list_index = np.random.choice(range(len(visit)), size = 5) #随机选择的索引，每行选择5个
        tourna_fit = np.array([fits[k] for k in tourna_list_index]) #提取
        min_fit = np.min(tourna_fit)
        min_list = visit[np.argmin(tourna_fit)] #选取适应度最小的访问顺序

        new_pops[i] = min_list #锦标赛后的子代适应度函数
        new_fits[i] = min_fit #锦标赛后的子代适应度函数

    return new_pops, new_fits

#单次迭代，进行单次的选择，交叉，变异
def Selection(visit, fits, distance_matrix): #输入未解码的值

    pops_1, fits_1 = tournament_select(visit, fits)
    pops_2, fits_2 = tournament_select(visit, fits)
    new_pops, new_fits = Cross(pops_1, pops_2, distance_matrix) #交叉
    new_pops, new_fits = Mutation(new_pops, new_fits, distance_matrix) #变异

    #一对一竞争
    for i in range(len(visit)):
        if fits[i] > new_fits[i]:
            fits[i] = new_fits[i]
            visit[i] = new_pops[i]

    return visit, fits

#主函数，进行TSP问题的求解
def choose(map):
    #GA
    popusize = 100  # 个体数
    generation = 500  # 迭代次数
    fits = [None] * popusize  # 适应度函数
    best_fit_list = np.array([])  # 存储每一代最优适应度
    city_num = len(map)  # 城市数量
    visit = np.array([np.random.choice([i for i in list(range(city_num))], city_num, replace=False)
                      for j in range(popusize)])  # 随机访问顺序
    distance_matrix = get_distance(map)  # 距离矩阵
    for i in range(popusize):
        fits[i] = sum_distance(visit[i], distance_matrix)  # 初代适应度
    best_fit = np.min(fits)
    best_fit_index = np.argmin(fits)
    best_fit_list = np.append(best_fit_list, best_fit)  # 添加到序列
    best_visit_list = visit[best_fit_index] #初代最优访问序列

    for iteration in range(generation):
        visit, fits = Selection(visit, fits, distance_matrix) #进行一次选择，交叉，变异
        best_fit = np.min(fits) #每一代最优
        best_fit_index = np.argmin(fits)
        best_fit_list = np.append(best_fit_list, best_fit)
        best_visit_list = np.append(best_visit_list, visit[best_fit_index])
        print('This is {} episode'.format(iteration))

    best_visit_list = best_visit_list.reshape(-1, city_num) #调整数组大小
    best_fit_index = np.argmin(best_fit_list)
    best_visit = best_visit_list[best_fit_index]
    PlotVisit(map, best_visit, np.min(best_fit_list))
    PlotFits(best_fit_list)

    return 0 #返回最佳轨迹与最短距离

if __name__ == '__main__':
    map = create_map()  # 获得地图数据
    choose(map) #问题求解
    plt.show()
