from cmath import inf
import numpy as np
import copy
import random
import heapq

import PkgAlg
import pkgMethod
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import edge
#import util
from util import *

"""
SkillRecord记录每一代中，每个任务的最小值
"""
class HelperRecord(object):
    def __init__(self, func_idx):
        self.func_idx = func_idx
        #当前代的信息
        self.gen_idx = -1
        self.min = inf
        self.avg = inf
        self.phen = None
        #历史最低值
        self.his_min = inf
        #历史最低值所在代数
        self.his_min_gen = -1
        #历史最低平均值
        self.his_avg = inf
        #历史最低平均值所在代数
        self.his_avg_gen = -1
        #相比上一代的下降程度
        self.min_drop = 0 # 数值下降越大越好
        self.avg_drop = 0 # 数值下降越大越好
        self.min_drop_rate = 0
        self.list_gen_min = [] #每一代中找到的最低值
        self.list_gen_avg = [] #每一代中找到的最低平均值

class AdaptiveHelper(object):
    def __init__(self, fit_func_list):
        #包含一组SkillRecord
        self.fit_func_list = fit_func_list

        #初始化HelperRecord
        self.dict_func = {}

        fit_func_idx = range(len(fit_func_list)+1)
        for func_idx in fit_func_idx:
            self.dict_func[func_idx] = HelperRecord(func_idx)
            #初始化时，没有第几代的信息
            #print(f'init {func_idx}')
            self.dict_func[func_idx].gen_idx = -1

    def add(self, func_idx, gen_idx, min, avg, phen):
        #print(f'add func_idx {func_idx},  gen_idx {gen_idx}, min {min} ')
        #如果不是第一代，则计算最低值的下降程度
        if self.dict_func[func_idx].gen_idx > 0:
            #如果小于最小值，则填入信息
            if min < self.dict_func[func_idx].min:
                self.dict_func[func_idx].gen_idx = gen_idx
                self.dict_func[func_idx].min = min
                self.dict_func[func_idx].avg = avg
                self.dict_func[func_idx].phen = phen
                #
                self.dict_func[func_idx].list_gen_min.append(min)
                self.dict_func[func_idx].list_gen_avg.append(avg)
            else:
                #如果不是最小值，则填入旧的最小值
                self.dict_func[func_idx].list_gen_min.append(self.dict_func[func_idx].min)
                self.dict_func[func_idx].list_gen_avg.append(self.dict_func[func_idx].avg)
            #计算下降程度
            self.dict_func[func_idx].min_drop = self.dict_func[func_idx].list_gen_min[-2] - self.dict_func[func_idx].list_gen_min[-1]
            self.dict_func[func_idx].avg_drop = self.dict_func[func_idx].list_gen_avg[-2] - self.dict_func[func_idx].list_gen_avg[-1]
            self.dict_func[func_idx].min_drop_rate = (self.dict_func[func_idx].list_gen_min[-2] - self.dict_func[func_idx].list_gen_min[-1])/self.dict_func[func_idx].list_gen_min[-2]
            
        #如果是第一代，第一代索引是0，先填入当前代的信息
        else:
            self.dict_func[func_idx].gen_idx = gen_idx
            self.dict_func[func_idx].min = min
            self.dict_func[func_idx].avg = avg
            self.dict_func[func_idx].phen = phen
            #
            self.dict_func[func_idx].list_gen_min.append(min)
            self.dict_func[func_idx].list_gen_avg.append(avg)

    def get_adaptive_helper(self, old_helper_idx):
        #如果是main task, 直接返回不修改
        if old_helper_idx == 0:
            return old_helper_idx
        #从helper list中找到min_drop程度最大的，以对应的大概率返回
        fit_func_idx = range(len(self.fit_func_list)+1)
        min_drop_rate_list = []
        for func_idx in fit_func_idx:
            min_drop_rate_list.append(self.dict_func[func_idx].min_drop_rate)
        #min_drop_rate_list数量与helper数量相同
        #min_drop_rate_list开始会是[0, 0, 0, 0]，第一个是main，后三个才是helper
        min_drop_rate_list_helper = min_drop_rate_list[1:]
        #一般是[0.0, 0.0, 0.0]返回0  [0.0, 0.0, 0.5]  [0.15, 0, 0.15] [0.2, 0, 0.06]
        #找到最大值并返回索引
        max_idx = np.argmax(min_drop_rate_list_helper)
        #以0.9的概率返回最大值，0.1的概率返回随机值
        rand = random.random()
        if rand < 0.9:
            #如果最大值是0
            if min_drop_rate_list_helper[max_idx] == 0:
                apative_helper_idx = old_helper_idx
            else:
                apative_helper_idx = max_idx+1
        else:
            #返回随机值
            idxs = range(len(min_drop_rate_list_helper))
            apative_helper_idx = random.choice(idxs) + 1

        
        return apative_helper_idx


def f1(query, arg_vec):
    config = query.config
    tmpScene = copy.deepcopy(query.scene)    
    newScene = edge.updateCurrentScene(config, tmpScene,arg_vec)
    cmni,cmpt,miga = edge.computeScene(config, newScene)

    latency = edge.statisticLatency(cmni,cmpt,miga)
    query.cur_fit_idx += 1

    fit_val = latency
    p = arg_vec
    if(query.cur_fit_idx == 1):
        query.result.best_fit_idx = query.cur_fit_idx
        query.result.best_fit_val = fit_val
        query.result.best_fit_phen = p
        query.result.add_fit_result(query.result.best_fit_idx, query.result.best_fit_val, query.result.best_fit_phen)

    if(fit_val < query.result.best_fit_val):
        query.result.best_fit_idx = query.cur_fit_idx
        query.result.best_fit_val = fit_val
        query.result.best_fit_phen = p
        query.result.add_fit_result(query.result.best_fit_idx, query.result.best_fit_val, query.result.best_fit_phen)

    query.result.add_all_fit_result(query.cur_fit_idx, query.result.best_fit_val, query.result.best_fit_phen)

    return latency

def f2(query, arg_vec):
    mask = query.user_group[0]
    arg_vec = mask_vec(mask, arg_vec)

    config = query.config
    tmpScene = copy.deepcopy(query.scene)    
    newScene = edge.updateCurrentScene(config, tmpScene,arg_vec)
    cmni,cmpt,miga = edge.computeScene(config, newScene)

    #energy = edge.computeEnergyWithPhen(config, arg_vec)
    query.cur_fit_idx += 1

    #开始选出最优值
    fit_val = cmni
    p = arg_vec
    if(query.cur_fit_idx == 2):
        query.result.f2_best_fit_idx = query.cur_fit_idx
        query.result.f2_best_fit_val = fit_val
        query.result.f2_best_fit_phen = p
        query.result.add_fit_result_f2(query.result.f2_best_fit_idx, query.result.f2_best_fit_val, query.result.f2_best_fit_phen)

    if(fit_val < query.result.f2_best_fit_val):
        query.result.f2_best_fit_idx = query.cur_fit_idx
        query.result.f2_best_fit_val = fit_val
        query.result.f2_best_fit_phen = p
        query.result.add_fit_result_f2(query.result.f2_best_fit_idx, query.result.f2_best_fit_val, query.result.f2_best_fit_phen)

    query.result.add_all_fit_result_f2(query.cur_fit_idx, query.result.f2_best_fit_val, query.result.f2_best_fit_phen)

    return cmni

def f3(query, arg_vec):
    mask = query.user_group[1]
    arg_vec = mask_vec(mask, arg_vec)

    config = query.config
    tmpScene = copy.deepcopy(query.scene)    
    newScene = edge.updateCurrentScene(config, tmpScene,arg_vec)
    cmni,cmpt,miga = edge.computeScene(config, newScene)

    #energy = edge.computeEnergyWithPhen(config, arg_vec)
    query.cur_fit_idx += 1

    #开始选出最优值
    """
    fit_val = cmni
    p = arg_vec
    if(query.cur_fit_idx == 2):
        query.result.f2_best_fit_idx = query.cur_fit_idx
        query.result.f2_best_fit_val = fit_val
        query.result.f2_best_fit_phen = p
        query.result.add_fit_result_f2(query.result.f2_best_fit_idx, query.result.f2_best_fit_val, query.result.f2_best_fit_phen)

    if(fit_val < query.result.f2_best_fit_val):
        query.result.f2_best_fit_idx = query.cur_fit_idx
        query.result.f2_best_fit_val = fit_val
        query.result.f2_best_fit_phen = p
        query.result.add_fit_result_f2(query.result.f2_best_fit_idx, query.result.f2_best_fit_val, query.result.f2_best_fit_phen)

    query.result.add_all_fit_result_f2(query.cur_fit_idx, query.result.f2_best_fit_val, query.result.f2_best_fit_phen)
    """
    return cmpt

def f4(query, arg_vec):
    mask = query.user_group[2]
    arg_vec = mask_vec(mask, arg_vec)

    config = query.config
    tmpScene = copy.deepcopy(query.scene)    
    newScene = edge.updateCurrentScene(config, tmpScene,arg_vec)
    cmni,cmpt,miga = edge.computeScene(config, newScene)

    #energy = edge.computeEnergyWithPhen(config, arg_vec)
    query.cur_fit_idx += 1

    #开始选出最优值
    """
    fit_val = cmni
    p = arg_vec
    if(query.cur_fit_idx == 2):
        query.result.f2_best_fit_idx = query.cur_fit_idx
        query.result.f2_best_fit_val = fit_val
        query.result.f2_best_fit_phen = p
        query.result.add_fit_result_f2(query.result.f2_best_fit_idx, query.result.f2_best_fit_val, query.result.f2_best_fit_phen)

    if(fit_val < query.result.f2_best_fit_val):
        query.result.f2_best_fit_idx = query.cur_fit_idx
        query.result.f2_best_fit_val = fit_val
        query.result.f2_best_fit_phen = p
        query.result.add_fit_result_f2(query.result.f2_best_fit_idx, query.result.f2_best_fit_val, query.result.f2_best_fit_phen)

    query.result.add_all_fit_result_f2(query.cur_fit_idx, query.result.f2_best_fit_val, query.result.f2_best_fit_phen)
    """
    return miga


def mask_vec(mask, vec):
    for idx, v in enumerate(vec):
        if idx in mask:
            vec[idx] = 0
    return vec

def get_ideal_features(query, scene):
    config = query.config
    ideal_hop = 0
    x_inServCapa = scene[config._inServCapa]
    #print(f'x_inServCapa\n{x_inServCapa}')
    x_inServPflNum = scene[config._inServPflNum]
    #print(f'x_inServPflNum\n{x_inServPflNum}')
    # if x_inServPflNum is 0, convert to 1 for representing all capa
    x_inServPflNum[np.where(x_inServPflNum == 0)] = 1
    #print(f'x_inServPflNum\n{x_inServPflNum}')
    x_serv_capa = x_inServCapa/x_inServPflNum
    #print(f'x_serv_capa\n{x_serv_capa}')
    ideal_serv_capa = max(x_serv_capa)
    #print(f'ideal_serv_capa\n{ideal_serv_capa}')
    x_congestion =  scene[config._userQueuLast] + scene[config._userQueu]
    #print(f'x_congestion\n{x_congestion}')
    ideal_congestion = min(x_congestion)
    #print(f'ideal_congestion\n{ideal_congestion}')
    #input()
    return ideal_hop, ideal_serv_capa, ideal_congestion

def generateCommunity(query, arg_vec):
    config = query.config
    #print(arg_vec)
    tmpScene = copy.deepcopy(query.scene)
    #print(f'scene\n{tmpScene}')
    scene = edge.updateCurrentScene(config, tmpScene,arg_vec)

    # compute each user's hop, servCapa/UserNum, queue+lastQueue

    dictUserServ = {}
    userCount = config.userCount #globalVar.get_value(enumVar.userCount)
    user = scene[config._userId,:]
    inCell = scene[config._uInCell,:]

    allServUserNum = edge.staAllServUserNum(config, scene)
    # if put user profile in this serv, so add 1
    allServUserNum = allServUserNum + 1
    #print(allServUserNum)
    allServQueue = edge.staAllServQueue(config, scene)
    #print(f'scene\n{scene}')
    #print(f'allServUserNum\n{allServUserNum}')
    #print(f'allServQueue\n{allServQueue}')
    #input()
    ideal_a1, ideal_a2, ideal_a3 = get_ideal_features(query, scene)

    for idxUser in range(userCount):
        x = inCell[idxUser]
        similarity = np.zeros(config.servCount)
        candiList = []
        for idxServ in range(config.servCount):
            # if put user profile in serv idxServ
            s = idxServ+1 # scene[config._pInServ, idxUser]
            #print(f'x {x}, s {s}')
            a1 = edge.getCellDistance(config, x, s )
            a2 = edge.getServCapa(config, s)/allServUserNum[idxServ]
            a3 = allServQueue[idxServ]
            #print(f'a1:{a1}, a2:{a2}, a3:{a3}')
            similarity[idxServ] = edge.get_similarity(
                ideal_a1, ideal_a2, ideal_a3, a1, a2, a3)
            if similarity[idxServ] > 0.99:
                candiList.append(s)
        if len(candiList) == 0:
            topIdx = np.argmax(similarity)
            candiList.append(topIdx+1)
        # in similarity array, find the top 3 serv
        dictUserServ[idxUser+1] = candiList


    return dictUserServ

def gen_neighbour(query, vec):
        dictUserServ = generateCommunity(query, vec)
        #print(dictUserServ)

        neighbour = []
        neighbour_num = int(1/query.pl) #self.cell_id_max - self.cell_id_min

        point_loc = np.random.randint(0, query.user_count)
        #point_val = vec[point_loc]

        #print(f'{vec}\npoint_loc:{point_loc}')

        _pInServ = query.config._pInServ #globalVar.get_value(enumScene._pInServ)
        neighbour.append(query.scene[_pInServ,:])

        travl = 0
        while(1):
            if len(neighbour) > neighbour_num:
                break

            if travl > query.user_count:
                break

            if point_loc > query.user_count-1:
                point_loc = 0

            community = dictUserServ[point_loc+1]
            #print(f'point_loc:{point_loc} com: {community}')
            # community has overlap loc
            used = []
            for c in community:
                if c in used:
                    continue
                new_vec = copy.deepcopy(vec)
                new_vec[point_loc] = c
                used.append(c)
                neighbour.append(new_vec)
                if len(neighbour) > neighbour_num:
                    break

            point_loc += 1
            travl += 1

        if len(neighbour) > neighbour_num:
            neighbour = neighbour[:neighbour_num]

        if len(neighbour) == 0:
            print('generated neighborhood is null!')
            neighbour.append(vec)
        

        neighbour = np.array(neighbour)
        return neighbour

def memetic(query, first_phen, first_val, population, fit_list):

    meme_num = int(query.pop_size * query.pl)
    meme_range = np.arange(query.pop_size)
    meme_idx = list(map(fit_list.index, heapq.nsmallest(meme_num,fit_list)))

    for i in meme_idx:
        arg_vec = population[i]
        arg_val = fit_list[i]

        neighbour = gen_neighbour(query, first_phen)
        len_nei = len(neighbour)

        # hill climbing local search
        start_num = int(meme_num/2)
        for start in range(start_num):
            step = 0
            start_point = np.random.randint(1, len_nei)
            if start == 0:
                    start_point = 0
            while(1): 
                step += 1
                start_point += 1

                if step >= len_nei:
                    break
                if start_point >= len_nei:
                    break


                nei_val = f1(query, neighbour[start_point])
                if nei_val < first_val:
                    first_phen = neighbour[start_point]
                    first_val = nei_val
                else:
                    break
                #print(f'meme idx {i}, new value:{self.fit_list[i]}')
            #util.sta_show_used_time_msg(start2, msg='hill climb')
    #query.parent = copy.deepcopy(population)
    #first_idx = np.argmin(fit_list)
    #first_fit = fit_list[first_idx]
    #first_phen = population[first_idx]

    return first_phen, first_val

def run_Single_MFEA_adaptive_meme2(query):
    #包括main和helper，f1是main
    fit_func_list = [f1, f2, f3, f4]
    #start = util.sta_show_now_time_msg(msg='begin a MFEA_adaptive_meme ')
    run_single_mfea_adapt_meme2(query, fit_func_list)
    #util.sta_show_used_time_msg(start, msg='')

def run_single_mfea_adapt_meme2(query, fit_func_list):

    # 评价函数列表
    functions = fit_func_list
    helper_func_list = fit_func_list[1:]

    # init random cros
    sta_helper = {}
    num_fit = len(fit_func_list)
    num_helper = len(helper_func_list)

    fit_func_idx = range(num_fit)
    helper_func_idx = range(num_helper)

    #switch_helper: 把helper_list列表转换为"索引-名称"键值对
    switch_helper = dict(zip(helper_func_idx, helper_func_list))
    # init sta_helper= 0
    for c in helper_func_idx:
        sta_helper[c] = 0
    # init end

    # 评价表，行是迭代次数，列是skill factor
    # 在每一代中，记录每个任务的最低值，平均值，phen
    fit_func_record = AdaptiveHelper(helper_func_list)


    query.cur_fit_idx = 0
    query.cur_gen_idx = 0
    result = query.result

    scene = query.scene
    user_count = query.user_count
    pop_size = query.pop_size
    max_gen = query.max_gen
    max_fit = query.max_fit
    pc = query.pc
    pm = query.pm
    pl = query.pl
    cell_id_min =  query.cell_id_min
    cell_id_max = query.cell_id_max

    

    # convert to mfea parameters
    K = num_fit
    # mfea中N = pop_size * K，表示f1和f2两个函数。
    #N = pop_size * K
    #但是因main+helper函数数量多，所以K设置为1
    N = pop_size
    D = user_count
    T = max_gen
    sbxdi = 10
    #pmdi  = 10
    pswap = 0.5
    rmp   = 0.3
    mutate = util.mutate
    variable_swap = util.variable_swap
    calculate_scalar_fitness = util.calculate_scalar_fitness
    get_optimization_results = util.get_optimization_results
    pmdi = query.pm

    # initialize
    low = query.cell_id_min
    high = query.cell_id_max+1
    # 设置为两代的种群数量
    population = np.random.randint(low, high, (2 * N, D) )
    #每个染色体使用哪个评价函数，skill_factor是用0和1交替设置初始值，2N行，一列
    skill_factor = np.array([i % K for i in range(2 * N)])
    #每个染色体在对应评价函数列的评价值，2N行，两列
    factorial_cost = np.full([2 * N, K], np.inf)
    #2N行，一列
    scalar_fitness = np.empty([2 * N])
    #每个染色体使用哪个task，2N行，一列
    helper_indicator = np.array([i % num_fit for i in range(2 * N)])

    # evaluate
    for i in range(2 * N):
        sf = skill_factor[i]
        factorial_cost[i, sf] = functions[sf](query, population[i])
    #关键的一步，对整个factorial_cost求scalar_fitness
    scalar_fitness = calculate_scalar_fitness(factorial_cost)

    # sort 
    sort_index = np.argsort(scalar_fitness)[::-1]
    population = population[sort_index]
    skill_factor = skill_factor[sort_index]
    factorial_cost = factorial_cost[sort_index]
    helper_indicator = helper_indicator[sort_index]

    # evolve
    #iterator = trange(T)
    iterator = range(T)
    for t in iterator:
        #start_iteration = util.sta_show_now_time_msg(msg='begin an iteration ')
        query.cur_gen_idx += 1
        if query.cur_fit_idx > max_fit:
            break
        # permute current population
        permutation_index = np.random.permutation(N)
        population[:N] = population[:N][permutation_index]
        skill_factor[:N] = skill_factor[:N][permutation_index]
        factorial_cost[:N] = factorial_cost[:N][permutation_index]
        factorial_cost[N:] = np.inf
        helper_indicator[:N] = helper_indicator[:N][permutation_index]
        transferred = np.zeros(2*N)

        # select pair to crossover
        for i in range(0, N, 2):
            
            p1, p2 = population[i], population[i + 1]
            sf1, sf2 = skill_factor[i], skill_factor[i + 1]
            ci1, ci2 = helper_indicator[i], helper_indicator[i + 1]

            # helper
            # set adaptive helper
            ci_active = random.sample(helper_func_idx, 1)[0]
            helper_name = switch_helper[ci_active]
            #crossover = globals().get('x_%s' % helper_name)
            #取消动态设置交叉算子
            crossover = util.x_single_point
            #print(f'ci_active {ci_active}')
            #print(f'{helper_name}')
            #print(crossover)
            #input()
            sta_helper[ ci_active ] += 1
            # set end
            #c1, c2 = crossover(p1, p2, _sbxdi=sbxdi, _pc=pc)
            if sf1 == sf2:
                c1, c2 = crossover(p1, p2, sbxdi, _pc=pc)
                c1 = mutate(query, c1)
                c2 = mutate(query, c2)
                c1, c2 = variable_swap(c1, c2, pswap)
                skill_factor[N + i] = sf1
                skill_factor[N + i + 1] = sf1

                # set adaptive crossover
                helper_indicator[N + i] = ci1
                helper_indicator[N + i + 1] = ci2
                # set end
            elif sf1 != sf2 and np.random.rand() < rmp:
                c1, c2 = crossover(p1, p2, sbxdi, _pc=pc)
                c1 = mutate(query, c1)
                c2 = mutate(query, c2)
                # c1, c2 = variable_swap(c1, c2, pswap)
                if np.random.rand() < 0.5: 
                    skill_factor[N + i] = sf1
                else: 
                    skill_factor[N + i] = sf2
                if np.random.rand() < 0.5: 
                    skill_factor[N + i + 1] = sf1
                else: 
                    skill_factor[N + i + 1] = sf2
                # set adaptive crossover
                helper_indicator[N + i] = ci_active
                helper_indicator[N + i + 1] = ci_active
                # set end
                # mark c1, c2 as transferred offspring
                transferred[N + i] = 1
                transferred[N + i + 1] = 1
            else:
                p2  = util.find_relative(population, skill_factor, sf1, N)
                c1, c2 = crossover(p1, p2, sbxdi, _pc=pc)
                c1 = mutate(query, c1)
                c2 = mutate(query, c2)
                c1, c2 = variable_swap(c1, c2, pswap)
                skill_factor[N + i] = sf1
                skill_factor[N + i + 1] = sf1
                # set adaptive crossover
                helper_indicator[N + i] = ci1
                helper_indicator[N + i + 1] = ci2
                # set end

            population[N + i, :], population[N + i + 1, :] = c1[:], c2[:]
        
        # evaluate
        for i in range(N, 2 * N):
            #增加动态调整helper
            #如果进入第二代后，第一代索引是0
            if t > 0:
                adaptive_helper = fit_func_record.get_adaptive_helper(skill_factor[i])
                skill_factor[i] = adaptive_helper
            sf = skill_factor[i]
            factorial_cost[i, sf] = functions[sf](query, population[i])
        scalar_fitness = calculate_scalar_fitness(factorial_cost)

        # sort
        sort_index = np.argsort(scalar_fitness)[::-1]
        population = population[sort_index]
        skill_factor = skill_factor[sort_index]
        factorial_cost = factorial_cost[sort_index]
        scalar_fitness = scalar_fitness[sort_index]

        c1 = population[np.where(skill_factor == 0)][0]
        c2 = population[np.where(skill_factor == 1)][0]

        #需要找出每个skill_factor中的最优phen
        for sk in range(num_fit):
            idx_list = np.where(skill_factor == sk)[0]
            #print(idx_list)
            #print(skill_factor[idx_list]) # [0 0 0] 都是0，或都是1，都是一样的sk
            #print(factorial_cost[idx_list, sk]) # [[7.32 8.01 8.12]] 从低到高排序，所以第一个是最优值

            sk_idx_first = idx_list[0]
            sk_val_list = factorial_cost[idx_list, sk]
            sk_min = sk_val_list[0]
            sk_avg = np.mean(sk_val_list)
            sk_phen = population[sk_idx_first]
            #sk是第几个fitness函数，t是第几代
            fit_func_record.add(sk, t, sk_min, sk_avg, sk_phen)

        #
        #print(fit_func_record.dict_func[0].list_gen_min)
        #input()


        # adaptation of transfer crossover indicators
        IR_best = None
        ci_best = None
        for i in range(0, N, 1):
          if transferred[N + i] == 1:
            #transferred[N + i + 1] = 1
            f_s = scalar_fitness[N + i]
            f_p = scalar_fitness[i]
            IR = (f_s - f_p)/abs(f_p)
            if IR_best == None:
              IR_best = IR
              ci_best = helper_indicator[N + i]
            else:
              if IR > IR_best:
                IR_best = IR
                ci_best = helper_indicator[N + i]
        for i in range(0, N, 1):
          if transferred[N + i] == 1:
            helper_indicator[N + i] = ci_best
          else:
            helper_indicator[N + i] = random.sample(fit_func_idx, 1)[0]

        # optimization info
        message = {'algorithm': 'mfea', 'rmp':rmp}
        results = get_optimization_results(t, population, factorial_cost, scalar_fitness, skill_factor, message)
        desc = 'gen:{} fitness:{} message:{}'.format(t, ' '.join('{:0.6f}'.format(res.fun) for res in results), message)
        ret = [res.fun for res in results]

        # statistic info
        gen_idx = query.cur_gen_idx
        first_phen = results[0].x
        first_val = results[0].fun
        #gen_idx, first_val, first_phen = self.select_offspring()

        #memetic
        #start = util.sta_show_now_time_msg(msg='begin a memetic ')
        #factorial_cost[:, 0]是主任务的fitness value col
        first_phen, first_val = memetic(query, first_phen, first_val,
        population, list(factorial_cost[:, 0]))
        #util.sta_show_used_time_msg(start, msg='')

        if(query.cur_gen_idx == 1):
            query.result.best_gen_idx = gen_idx
            query.result.best_gen_val = first_val
            query.result.best_gen_phen = first_phen
            query.result.add_gen_result(query.result.best_gen_idx, 
                                        query.result.best_gen_val, 
                                        query.result.best_gen_phen)

        if(first_val < query.result.best_gen_val):
            query.result.best_gen_idx = gen_idx
            query.result.best_gen_val = first_val
            query.result.best_gen_phen = first_phen
            query.result.add_gen_result(query.result.best_gen_idx, 
                                        query.result.best_gen_val, 
                                        query.result.best_gen_phen)

        query.result.add_all_gen_result(query.cur_gen_idx, 
                                        query.result.best_gen_val, 
                                        query.result.best_gen_phen)
        #util.sta_show_used_time_msg(start_iteration, msg='')
    return query.result
