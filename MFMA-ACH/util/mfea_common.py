import numpy as np
from copy import deepcopy
from scipy.optimize import OptimizeResult
import random

def calculate_scalar_fitness(factorial_cost):
    '''
    print(f'factorial_cost:{factorial_cost.shape}\n{factorial_cost}')
    temp1 = np.argsort(factorial_cost, axis=0)
    print(f'temp1:{temp1.shape}\n{temp}')
    temp2 = np.argsort(temp1, axis=0)
    print(f'temp2:{temp2.shape}\n{temp2}')
    temp3 = np.min(temp2+1, axis=1)
    print(f'temp3:{temp3.shape}\n{temp3}')
    '''
    #factorial_cost是2N行，两列（等于评价函数的数量）
    #axis=0沿着行向下(每列)的元素进行排序
    ret = 1 / np.min(np.argsort(np.argsort(factorial_cost, axis=0), axis=0) + 1, axis=1)
    #print(f'ret：{ret.shape}\n{ret}')
    #input()
    return ret


def calculate_scalar_fitness_multi(factorial_cost_multi, sf, O):
    # factorial_cost_multi (row: pop_size, col: 4, col 0:)
    f1_col_1 = factorial_cost_multi[:,0]
    f1_col_2 = factorial_cost_multi[:,2]

    f2_col_1 = factorial_cost_multi[:,1]
    f2_col_2 = factorial_cost_multi[:,3]
    
    row = factorial_cost_multi.shape[0]
    idx_array = np.arange(row)

    distance_arr_f1, rank_score_f1 = rank_distance(idx_array, f1_col_1, f1_col_2)
    distance_arr_f2, rank_score_f2 = rank_distance(idx_array, f2_col_1, f2_col_2)

    rank_score_f1 = rank_score_f1.reshape((-1, 1))
    rank_score_f2 = rank_score_f2.reshape((-1, 1))
    factorial_combine = np.hstack((rank_score_f1, rank_score_f2))

    factorial_cost = calculate_scalar_fitness(factorial_combine)

    return factorial_cost


def rank_distance(idx_array, f1_array, f2_array):
    distance_list = []
    point1, point2 = select_extreme_points(f1_array, f2_array)
    for i in idx_array:
        point = np.array([f1_array[i], f2_array[i]])
        distance_list.append(point_distance_line(point, point1, point2))
    index_score = np.argsort(-np.array(distance_list))
    rank_score = np.argsort(index_score) #+1
    distance_arr = np.array(distance_list)
    return distance_arr, rank_score

def select_extreme_points(f1_array_ex, f2_array_ex):
    # delete inf element
    f1_array = deepcopy(f1_array_ex)
    f2_array = deepcopy(f2_array_ex)
    f1_array[f1_array == float("inf")] = float("-inf")
    f2_array[f2_array == float("inf")] = float("-inf")

    far_f1_idx = np.argmax(f1_array)
    far_f2_idx = np.argmax(f2_array)
    new_point_flag = False
    if far_f1_idx == far_f2_idx:
        #print(f'\nextreme points are the same, in mfea_common.')
        combine = np.hstack((np.arange(len(f1_array)).reshape(-1,1),f1_array.reshape(-1,1),f2_array.reshape(-1,1)))
        #print(f'the same index: {far_f1_idx}\n{combine}')
        #input()
        tmp_f1 = deepcopy(f1_array)
        tmp_f2 = deepcopy(f2_array)
        allow = 2
        while(allow):
            allow = allow -1
            if far_f1_idx != far_f2_idx:
                break
            if allow == 1:
                new_point_flag = True
                break
            
            tmp_f1[far_f1_idx] = min(tmp_f1) - 1
            tmp_f2[far_f1_idx] = min(tmp_f2) - 1
            far_f1_idx = np.argmax(tmp_f1)
            far_f2_idx = np.argmax(tmp_f2)
            
            print(f'new {far_f1_idx} {far_f2_idx} ')
        #input()
    point1 = np.array([f1_array[far_f1_idx], f2_array[far_f1_idx]])
    point2 = np.array([f1_array[far_f2_idx], f2_array[far_f2_idx]])
    if new_point_flag:
        sigle_point = (f1_array[far_f1_idx], f2_array[far_f2_idx])
        zero_point = (0, 0)
    return point1, point2

def point_distance_line(point, line_point1, line_point2):
    #compute the distance from a point to a line
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
    return distance

def find_relative(population, skill_factor, sf, N):
    return population[np.random.choice(np.where(skill_factor[:N] == sf)[0])]

##################################################################
# new corssover operators
def x_single_point(p1, p2, _sbxdi=None, _pc=1.0):
    #交叉概率判断是否要做交叉
    randVar= np.random.rand()
    if randVar>_pc:
        return p1, p2

    #print('single_point_x')
    D = p1.shape[0]
    # Randomly generating one crossover point
    crossover_point = random.randint(1, D-2)

    c1 = deepcopy(p1)
    c2 = deepcopy(p2)
    # Crossing over from randomly generated point
    c1[crossover_point:] = p2[crossover_point:]
    c2[crossover_point:] = p1[crossover_point:]

    return c1, c2


def x_two_point(p1, p2, _sbxdi=None, _pc=1.0):

    #交叉概率判断是否要做交叉
    randVar= np.random.rand()
    if randVar>_pc:
        return p1, p2

    #print('two_point_x')
    D = p1.shape[0]
    # Randomly generating two crossover point
    cross_point_num = int(2)
    cross_range = np.arange(1,D-1)
    cross_point_idx = np.random.choice(cross_range, size=cross_point_num, replace=False)
    cross_point_idx.sort()
    point_1 = cross_point_idx[0]
    point_2 = cross_point_idx[1]

    c1 = deepcopy(p1)
    c2 = deepcopy(p2)
    # Crossing over between randomly generated points
    c1[point_1:point_2] = p2[point_1:point_2]
    c2[point_1:point_2] = p1[point_1:point_2]

    return c1, c2

def x_uniform_point(p1, p2, _sbxdi=None, _pc=1.0):
    #交叉概率判断是否要做交叉
    randVar= np.random.rand()
    if randVar>_pc:
        return p1, p2

    #print('uniform_point_x')
    D = p1.shape[0]
    c1 = deepcopy(p1)
    c2 = deepcopy(p2)

    cross_point_num = int(2)
    cross_range = np.arange(1,D-1)
    cross_point_idx = np.random.choice(cross_range, size=cross_point_num, replace=False)
    cross_point_idx.sort()
    point_1 = cross_point_idx[0]
    point_2 = cross_point_idx[1]

    c1 = deepcopy(p1)
    c2 = deepcopy(p2)
    # Crossing over between randomly generated points
    c1[point_1:point_2] = p2[point_1:point_2]
    c2[point_1:point_2] = p1[point_1:point_2]

    probability_of_swap = 0.8
    #for idx, gene in enumerate(p1):
    for idx in range(len(c1)):
        r = random.random()
        if r > probability_of_swap:
            c1[idx] = p2[idx]
            c2[idx] = p1[idx]
    return c1, c2

def x_complex_two_uniform(p1, p2, _sbxdi=None, _pc=1.0):
    #交叉概率判断是否要做交叉
    randVar= np.random.rand()
    if randVar>_pc:
        return p1, p2

    #print('uniform_point_x')
    D = p1.shape[0]
    # Randomly generating two crossover point
    cross_point_num = int(2)
    cross_range = np.arange(1,D-1)
    cross_point_idx = np.random.choice(cross_range, size=cross_point_num, replace=False)
    cross_point_idx.sort()
    point_1 = cross_point_idx[0]
    point_2 = cross_point_idx[1]

    c1 = deepcopy(p1)
    c2 = deepcopy(p2)
    # Crossing over between randomly generated points
    c1[point_1:point_2] = p2[point_1:point_2]
    c2[point_1:point_2] = p1[point_1:point_2]

    probability_of_swap = 0.9
    #for idx, gene in enumerate(p1):
    for idx in range(len(p1)):
        r = random.random()
        if r > probability_of_swap:
            temp = c2[idx]
            c1[idx] = c2[idx]
            c2[idx] = temp
    return c1, c2

def x_arithmetic(p1, p2, _sbxdi=None, _pc=None):
    #交叉概率判断是否要做交叉
    randVar= np.random.rand()
    if randVar>_pc:
        return p1, p2

    # float
    if _pc == None:
        pc = 0.6
    else:
        pc = _pc

    D = p1.shape[0]
    c1 = deepcopy(p1)
    c2 = deepcopy(p2)
    #randVar= np.random.rand()
    #if randVar<pc:
    w=np.random.rand()
    for i in range(D):
        c1[i] = p1[i]*w + p2[i]*(1-w)
        c2[i] = p1[i]*(1-w) + p2[i]*w
    return c1, c2
    #else:
    #    return p1, p2

def x_geometrical(p1, p2, _sbxdi=None, _pc=None):
    # float    
    D = p1.shape[0]
    c1 = deepcopy(p1)
    c2 = deepcopy(p2)

    for i in range(D):
        c1[i] = np.sqrt(p1[i]*p2[i])  # == np.power(p1[i], 0.5)*np.power(p2[i], 0.5)
        randVar= np.random.rand()
        c2[i] = np.power(p1[i], randVar)*np.power(p2[i], 1-randVar)
    return c1, c2

def x_BLX_alpha(p1, p2, _sbxdi=None, _pc=None):
    # float    
    D = p1.shape[0]
    c1 = deepcopy(p1)
    c2 = deepcopy(p2)

    alpha = 0.5
    randVar= np.random.rand()
    gamma = (1+2*alpha)*randVar-alpha

    for i in range(D):
        c1[i] = (1-gamma)*p1[i] + gamma*p2[i]
        c2[i] = (1-gamma)*p2[i] + gamma*p1[i]
    return c1, c2

def x_cut_splice(p1, p2, _sbxdi=None, _pc=None):
    #交叉概率判断是否要做交叉
    randVar= np.random.rand()
    if randVar>_pc:
        return p1, p2

    #print('cut_splice_x')
    p1 = list(p1)
    p2 = list(p2)
    #D = p1.shape[0]
    D = len(p1)
    # Randomly generating two crossover point
    cross_point_num = int(2)
    cross_range = np.arange(1,D-1)
    cross_point_idx = np.random.choice(cross_range, size=cross_point_num, replace=False)
    cross_point_idx.sort()
    point_1 = cross_point_idx[0]
    point_2 = cross_point_idx[1]

    c1 = deepcopy(p1)
    c2 = deepcopy(p2)
    # Crossing over between randomly generated points
    c1[point_1:] = p2[point_2:]
    c2[point_2:] = p1[point_1:]

    # To ensure list is not smaller than D
    var1 = list((random.randint(1, D) for x in range(len(c1), D)))
    c1.extend(var1)
    var2 = list((random.randint(1, D) for x in range(len(c2), D)))
    c2.extend(var2)

    # To ensure list is not bigger than D
    del c1[D:]
    del c2[D:]

    c1 = np.array(c1)
    c2 = np.array(c2)
    return c1, c2

def xx(p1, p2):
    p1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    p2 = np.array([11, 22, 33, 44, 55, 66, 77, 88, 99])
    D = p1.shape[0]
    c1 = deepcopy(p1)
    c2 = deepcopy(p2)

    probability_of_swap = 0.5
    for idx, gene in enumerate(p1):
        r = random.random()
        print(r)
        if r > probability_of_swap:
            c1[idx] = p2[idx]
            c2[idx] = p1[idx]
    print(f'c1: {c1}')
    print(f'c2: {c2}')
    input()
    return c1, c2

#############################################

def mutate(query, phen):
    #多点变异
    #点的数量是len_phen * pm
    pm = query.pm
    ret_phen = deepcopy(phen)
    # mutation_point = np.random.randint(0,self.user_count)
    len_phen = len(phen)
    idx_list = np.arange(len_phen)
    muta_num = int(len_phen * pm)
    muta_in_chrom_idx = np.random.choice(idx_list, size=muta_num, replace=False)
    for y in muta_in_chrom_idx:
        mutation_context = np.random.randint(query.cell_id_min, query.cell_id_max + 1)
        # print(f'mutationPoint row:{x} col:{mutation_point} -> {mutation_context}')
        ret_phen[y] = mutation_context        
    return ret_phen

def mutate_single(query, phen):
    pm = query.pm

    # randVar= np.random.rand()
    # if randVar>pm:
    #     return phen
    
    ret_phen = deepcopy(phen)
    # mutation_point = np.random.randint(0,self.user_count)
    len_phen = len(phen)
    idx_list = np.arange(len_phen)
    muta_num = 1 #int(len_phen * pm)
    muta_in_chrom_idx = np.random.choice(idx_list, size=muta_num, replace=False)
    for y in muta_in_chrom_idx:
        mutation_context = np.random.randint(query.cell_id_min, query.cell_id_max + 1)
        # print(f'mutationPoint row:{x} col:{mutation_point} -> {mutation_context}')
        ret_phen[y] = mutation_context        
    return ret_phen

def mutate_uniform(query, phen):
    pm = query.pm

    # randVar= np.random.rand()
    # if randVar>pm:
    #     return phen

    ret_phen = deepcopy(phen)
    # mutation_point = np.random.randint(0,self.user_count)
    len_phen = len(phen)
    idx_list = np.arange(len_phen)
    #muta_num = 1 #int(len_phen * pm)
    #muta_in_chrom_idx = np.random.choice(idx_list, size=muta_num, replace=False)
    for y in idx_list:
        randVar= np.random.rand()
        if randVar>0.5:
            mutation_context = np.random.randint(query.cell_id_min, query.cell_id_max + 1)
            # print(f'mutationPoint row:{x} col:{mutation_point} -> {mutation_context}')
            ret_phen[y] = mutation_context
    return ret_phen

def mutate_swap(query, phen):
    pm = query.pm

    # randVar= np.random.rand()
    # if randVar>pm:
    #     return phen
    
    ret_phen = deepcopy(phen)
    # mutation_point = np.random.randint(0,self.user_count)
    len_phen = len(phen)
    idx_list = np.arange(len_phen)
    muta_num = 2 #int(len_phen * pm)
    muta_in_chrom_idx = np.random.choice(idx_list, size=muta_num, replace=False)

    idx1 = muta_in_chrom_idx[0]
    idx2 = muta_in_chrom_idx[1]
    temp = ret_phen[idx1]
    ret_phen[idx1] = ret_phen[idx2]
    ret_phen[idx2] = temp
    # for y in muta_in_chrom_idx:
        #mutation_context = np.random.randint(query.cell_id_min, query.cell_id_max + 1)
        # print(f'mutationPoint row:{x} col:{mutation_point} -> {mutation_context}')
        #ret_phen[y] = mutation_context      
        # temp = ret_phen[y]
    return ret_phen

#############################################


def variable_swap(p1, p2, probswap):
    #p1 = np.array([1, 2, 3, 4, 5])
    #p2 = np.array([6, 7, 8, 9, 0])
    D = p1.shape[0]
    swap_indicator = np.random.rand(D) <= probswap
    c1, c2 = p1.copy(), p2.copy()
    c1[np.where(swap_indicator)] = p2[np.where(swap_indicator)]
    c2[np.where(swap_indicator)] = p1[np.where(swap_indicator)]
    return c1, c2


def get_best_individual(population, factorial_cost, scalar_fitness, skill_factor, sf):
      # select individuals from task sf
      idx                = np.where(skill_factor == sf)[0]
      subpop             = population[idx]
      sub_factorial_cost = factorial_cost[idx]
      sub_scalar_fitness = scalar_fitness[idx]

      # select best individual
      idx = np.argmax(sub_scalar_fitness)
      x = subpop[idx]
      fun = sub_factorial_cost[idx, sf]
      return x, fun

def get_best_individual_multi(population, factorial_cost_multi, scalar_fitness, skill_factor, sf):
      # select individuals from task sf
      idx                = np.where(skill_factor == sf)[0]
      subpop             = population[idx]
      sub_factorial_cost = factorial_cost_multi[idx]
      sub_scalar_fitness = scalar_fitness[idx]

      # select best individual
      idx = np.argmax(sub_scalar_fitness)
      x = subpop[idx]
      f1 = sub_factorial_cost[idx, sf]
      f2 = sub_factorial_cost[idx, sf+2]
      return x, f1, f2

def get_optimization_results(t, population, factorial_cost, scalar_fitness, skill_factor, message):
    K = len(set(skill_factor))
    N = len(population) // 2
    results = []
    for k in range(K):
        result         = OptimizeResult()
        x, fun         = get_best_individual(population, factorial_cost, scalar_fitness, skill_factor, k)
        result.x       = x  #phen, the solution
        result.fun     = fun  #fitness, value of the objective function
        result.message = message
        result.nit     = t  #Number of iterations
        result.nfev    = (t + 1) * N  #Number of evaluations of the objective functions
        results.append(result)
    return results

def get_optimization_results_multi(t, population, factorial_cost_multi, scalar_fitness, skill_factor, message):
    K = len(set(skill_factor))
    N = len(population) // 2
    results = []
    for k in range(K):
        result         = OptimizeResult()
        x, f1, f2         = get_best_individual_multi(population, factorial_cost_multi, scalar_fitness, skill_factor, k)

        result.x       = x  #phen, the solution
        result.fun     = [f1, f2]  #fitness, value of the objective function
        result.message = message
        result.nit     = t  #Number of iterations
        result.nfev    = (t + 1) * N  #Number of evaluations of the objective functions
        results.append(result)
    return results