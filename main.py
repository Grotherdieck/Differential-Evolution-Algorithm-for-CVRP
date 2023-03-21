import pandas as pd
import random
import numpy as np

# 读取customer_data数据
# df1 = pd.read_excel('customer_data.xlsx', sheet_name='Sheet1')
df1 = pd.read_excel('customer_data_ramdom.xlsx', sheet_name='Sheet1')
# print(df)

# 把客户需求量存到一个列表customer_list中
customer_list = list()
for index, row in df1.iterrows():
    # print("Index: ", index)
    # print("Data: ", row[0], row[1], row[2])
    tmplist = list()
    for i in range(3):
        tmplist.append(row[i])
    customer_list.append(tmplist)
# print(customer_list)

# 把客户之间的距离distance.xlsx读到一个矩阵distance里
# df2 = pd.read_excel('distance.xlsx', sheet_name='Sheet1')
df2 = pd.read_excel('distance_random.xlsx', sheet_name='Sheet1')
distance = df2.values.tolist()
# print(distance)

# 相关参数设置
size=50      # 种群中的个体数量
cr_min=0.3   # 最小变异概率
cr_max=0.9   # 最大变异概率
n=200        # 迭代次数
L=400        # 车辆最大行驶距离
kmax=3       # 最多车辆数
# Q=8          # 车辆载重
Q=0
for seq in customer_list:
    Q = Q + seq[2]
Q = Q / (kmax - 1)
print(Q)

F=0.5        # 放缩因子
customer_size=len(customer_list) # 客户点数量
# print(customer_size)
punish_num = 10000 # 不满足约束的惩罚

#初始化种群 随机生成size个1-customer_size的排列 在计算时 需要开头补上0 结尾补上0 插入种群中
population_set = list()
for i in range(size):
    seq = list(range(1, customer_size + 1))
    random.shuffle(seq)
    # print(seq)
    population_set.append(seq)

# 获得某规划的distance
def GetDistance(schel_list):
    if not isinstance(schel_list, list):
        raise TypeError("vec must be a list")
    sum_of_distance = 0
    for seq in schel_list:
        sum_of_distance = sum_of_distance + distance[seq[-1]][0]
        seq_len = len(seq)
        for idx in range(0, seq_len):
            if (idx == 0):
                sum_of_distance = sum_of_distance + distance[0][seq[0]]
            else:
                sum_of_distance = sum_of_distance + distance[seq[idx - 1]][seq[idx]]
    return sum_of_distance

# 按和分割的函数
def SplitBySum(vec):
    if not isinstance(vec, list):
        raise TypeError("vec must be a list")
    # print(vec)
    # 根据客户取货数量和车辆容量限制做第一次分割
    cur_load = 0
    split_list = list()
    split_tmp = list()
    for pos in vec:
        cur_load = cur_load + customer_list[pos - 1][2]
        # print(pos)
        # print(cur_load, customer_list[pos - 1][2])
        if (cur_load <= Q):
            # print("here " + str(pos))
            split_tmp.append(pos)
            # print(split_tmp)
        else:
            split_list.append(split_tmp)
            split_tmp = list()
            cur_load = customer_list[pos - 1][2]
            split_tmp.append(pos)
    if (len(split_tmp) != 0):
        split_list.append(split_tmp)
    # print(split_list)
    return split_list

# 分割函数
def MySplit(vec):
    if not isinstance(vec, list):
        raise TypeError("vec must be a list")
    vec_size = len(vec)
    if (vec_size == 1):
        ans_list = list()
        ans_list.append(vec)
        return ans_list
    split_list_first = SplitBySum(vec)
    split_list_first_size = len(split_list_first)
    if (split_list_first_size == 0):
        ans_list = list()
        ans_list.append(vec)
        return ans_list
    cur_vec = split_list_first[0]
    cur_vec_size = len(cur_vec)
    init_cap = 0 # 计算开始的承重
    for obj in cur_vec:
        init_cap = init_cap + customer_list[obj - 1][2]
    # 运货量
    flag = False
    memory_idx = -1
    for i in range(0, cur_vec_size):
        # print("here" + str(cur_vec[i]))
        init_cap = init_cap - customer_list[cur_vec[i] - 1][2] + customer_list[cur_vec[i] - 1][1]
        if (init_cap > Q):
            flag = True
            memory_idx = i
            break
    # 距离
    total_distance = 0
    # total_distance = total_distance + distance[0][cur_vec[0]]
    for i in range(0, cur_vec_size):
        if (i == 0):
            total_distance = total_distance + distance[0][cur_vec[0]]
        else:
            total_distance = total_distance + distance[cur_vec[i - 1]][cur_vec[i]]
        if (total_distance > L):
            flag = True
            if (memory_idx == -1):
                memory_idx = i
            else:
                memory_idx = min(memory_idx, i)
            break
    total_distance = total_distance + distance[0][cur_vec[-1]]
    if (total_distance > L):
        flag = True
        if (memory_idx == -1):
            memory_idx = max(cur_vec_size - 2, 0)
        else:
            memory_idx = min(memory_idx, max(cur_vec_size - 2, 0))
    ans_list = list()
    if not flag:
        ans_list.append(cur_vec)
        next_vec = list()
        for i in range(1, len(split_list_first)):
            for obj in split_list_first[i]:
                next_vec.append(obj)
        # ans_list.append(MySplit(next_vec))
        for obj in MySplit(next_vec):
            if not (len(obj) == 0):
                ans_list.append(obj)
        return ans_list
    else:
        to_insert_vec = list()
        next_vec = list()
        # print(memory_idx)
        for i in range(0, cur_vec_size):
            # print("what " + str(cur_vec[i]))
            if (i <= memory_idx):
                to_insert_vec.append(cur_vec[i])
            else:
                next_vec.append(cur_vec[i])
        for i in range(1, len(split_list_first)):
            for obj in split_list_first[i]:
                next_vec.append(obj)
        ans_list.append(to_insert_vec)
        for obj in MySplit(next_vec):
            if not (len(obj) == 0):
                ans_list.append(obj)
        return ans_list

# def test(vec):
#     test_split_list = MySplit(vec)
#     print(test_split_list)

# 计算适应度的函数(已经弃用)
# def Adaptability(vec):
#     if not isinstance(vec, list):
#         raise TypeError("vec must be a list")
#     print(vec)
#     # 根据客户取货数量和车辆容量限制做第一次分割
#     cur_load = 0
#     split_list = list()
#     split_tmp = list()
#     for pos in vec:
#         cur_load = cur_load + customer_list[pos - 1][2]
#         # print(pos)
#         # print(cur_load, customer_list[pos - 1][2])
#         if (cur_load <= Q):
#             # print("here " + str(pos))
#             split_tmp.append(pos)
#             # print(split_tmp)
#         else:
#             split_list.append(split_tmp)
#             split_tmp = list()
#             cur_load = customer_list[pos - 1][2]
#             split_tmp.append(pos)
#     if (len(split_tmp) != 0):
#         split_list.append(split_tmp)
#     print(split_list)
#     ans = 0.0
#     if (len(split_list) > kmax):
#         print("This ans out of kmax.")
#         # return 1 / ((len(split_list) - kmax) * punish_num)
#         ans = ans + (len(split_list) - kmax) * punish_num
#     # 根据送货需求和取货需求计算split_list中每个个体是否还需要分割
#     second_split_list = list()
#     for seq in split_list:
#         # 前后补上0
#         cur_seq = list()
#         cur_seq.append(0)
#         cur_seq = cur_seq + seq
#         cur_seq.append(0)
#         # print(cur_seq)
#         # 取货送货看是否满足
#         init_cap = 0 # 计算开始的承重
#         for obj in seq:
#             init_cap = init_cap + customer_list[obj - 1][2]
        # print(init_cap)
        # for idx in seq:
        #     init_cap = init_cap - customer_list[idx - 1][2] + customer_list[idx - 1][1]
        #     if (init_cap > Q):

# 一些测试
# Adaptability(population_set[0])
# test(population_set[0])
# test([4, 2, 7, 5, 8, 6, 1, 3])
# test([8, 6, 1, 5, 2, 4, 7, 3])
# for populatin in population_set:
#     test(populatin)

# 计算适应度并且返回调度方案的函数
def Fitness(vec):
    if not isinstance(vec, list):
        raise TypeError("vec must be a list")
    my_schedul_list = MySplit(vec)
    # print(my_schedul_list)
    my_schedul_list_size = len(my_schedul_list)
    sum_of_distance = 0
    for seq in my_schedul_list:
        sum_of_distance = sum_of_distance + distance[seq[-1]][0]
        seq_len = len(seq)
        for idx in range(0, seq_len):
            if (idx == 0):
                sum_of_distance = sum_of_distance + distance[0][seq[0]]
            else:
                sum_of_distance = sum_of_distance + distance[seq[idx - 1]][seq[idx]]
    # print("distance is " + str(sum_of_distance))
    punish_cnt = my_schedul_list_size - kmax
    # if (punish_cnt == 0):
    #     print("success!")
    #     print(my_schedul_list)
    #     print(sum_of_distance)
    # print("Fitness = " + str(1 / (sum_of_distance + punish_num * punish_cnt)))
    return (1 / (sum_of_distance + punish_num * punish_cnt), my_schedul_list)
    # return (sum_of_distance + punish_num * punish_cnt, my_schedul_list)
# Fitness test
# Fitness([4, 2, 7, 5, 8, 6, 1, 3])
# for populatin in population_set:
#     Fitness(populatin)

# 保存全局最好个体及其解决方案
best_individual = list()
best_individual_fitness = 0.0
# def UpdateBestIndivial():
for population in population_set:
    curfitness, cur_list = Fitness(population)
    if (curfitness > best_individual_fitness):
        best_individual_fitness = curfitness
        best_individual = cur_list


# UpdateBestIndivial()

print("Inital fitness : " + str(best_individual_fitness))
print("Inital individual : ")
print(best_individual)

# 主循环
for iteration_cnt in range(1, n + 1):
    # 一、种群变异
    v_population_set = list() # 变异个体集合
    # 产生变异集合
    chrom_len = len(population_set[0])
    for chrom_c in population_set:
        cur_v_list = list()
        chrom_a = population_set[random.randrange(size)]
        chrom_b = population_set[random.randrange(size)]
        for i in range(0, chrom_len):
            cur_num = chrom_c[i] + F * (chrom_a[i] - chrom_b[i])
            cur_v_list.append(cur_num)
        v_population_set.append(cur_v_list)
    # 二、交叉操作
    # 交叉概率
    cr = cr_min + iteration_cnt * (cr_max - cr_min) / n
    trial_population_inital_set = list()
    for i in range(0, size):
        v = v_population_set[i]
        chrom = population_set[i]
        trial_list = list()
        for j in range(0, chrom_len):
            cur_rand = random.random()
            if (cur_rand <= cr):
                trial_list.append(v[j])
            else:
                trial_list.append(chrom[j])
        trial_population_inital_set.append(trial_list)
    # print(trial_population_inital_set)
    # 三、把trial_population_inital_set给标准化
    trial_population_set = list()
    for trial in trial_population_inital_set:
        sorted_indices = np.argsort(trial)
        ranked_list = [0] * len(trial)
        for rank, index in enumerate(sorted_indices, 1):
            ranked_list[index] = rank
        trial_population_set.append(ranked_list)
    # print(trial_population_set)
    # 四、选择并诞生新一代种群
    next_population = list()
    for idx in range(0, size):
        chrom_fitness, chrom_list = Fitness(population_set[idx])
        trial_fitness, trial_list = Fitness(trial_population_set[idx])
        if (trial_fitness <= chrom_fitness):
            next_population.append(population_set[idx])
        else: 
            next_population.append(trial_population_set[idx])
    population_set = next_population
    # 五、更新全局最优解
    for population in population_set:
        curfitness, cur_list = Fitness(population)
        if (curfitness > best_individual_fitness):
            best_individual_fitness = curfitness
            best_individual = cur_list
    print("当前是第" + str(iteration_cnt) + "轮, 当前最优解是:" + str(GetDistance(best_individual)))

print("best indivial : ")
print(best_individual)
print("best distance : " + str(GetDistance(best_individual)))
