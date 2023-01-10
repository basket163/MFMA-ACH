import numpy as np
import copy

import PkgAlg
import pkgMethod
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import edge
import util

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

def run_single_mfea_ii(query):
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

    functions = [f1, f2]

    # convert to mfea parameters
    K = 2  # number of functions
    N = pop_size * K
    D = user_count
    T = max_gen
    sbxdi = 10
    #pmdi  = 10
    pswap = 0.5
    # mfea-ii修改处，用rmp_matrix代替rmp
    #rmp   = 0.3
    rmp_matrix = np.zeros([K, K])
    mutate = util.mutate
    variable_swap = util.variable_swap
    calculate_scalar_fitness = util.calculate_scalar_fitness
    get_optimization_results = util.get_optimization_results
    pmdi = query.pm

    # initialize
    low = query.cell_id_min
    high = query.cell_id_max+1
    population = np.random.randint(low, high, (2 * N, D) )
    skill_factor = np.array([i % K for i in range(2 * N)])
    factorial_cost = np.full([2 * N, K], np.inf)
    scalar_fitness = np.empty([2 * N])

    # evaluate
    for i in range(2 * N):
        sf = skill_factor[i]
        factorial_cost[i, sf] = functions[sf](query, population[i])
    scalar_fitness = calculate_scalar_fitness(factorial_cost)

    # sort 
    sort_index = np.argsort(scalar_fitness)[::-1]
    population = population[sort_index]
    skill_factor = skill_factor[sort_index]
    factorial_cost = factorial_cost[sort_index]

    # evolve
    #iterator = trange(T)
    iterator = range(T)
    for t in iterator:
        query.cur_gen_idx += 1
        if query.cur_fit_idx > max_fit:
            break
        # permute current population
        permutation_index = np.random.permutation(N)
        population[:N] = population[:N][permutation_index]
        skill_factor[:N] = skill_factor[:N][permutation_index]
        factorial_cost[:N] = factorial_cost[:N][permutation_index]
        factorial_cost[N:] = np.inf

        #mfea-ii修改处
        #learn rmp
        subpops    = PkgAlg.get_subpops(population, skill_factor, N)
        rmp_matrix = PkgAlg.learn_rmp(subpops, D)


        # select pair to crossover
        for i in range(0, N, 2):
            p1, p2 = population[i], population[i + 1]
            sf1, sf2 = skill_factor[i], skill_factor[i + 1]

            # crossover
            if sf1 == sf2:
                c1, c2 = util.x_two_point(p1, p2, sbxdi)
                c1 = mutate(query, c1)
                c2 = mutate(query, c2)
                c1, c2 = variable_swap(c1, c2, pswap)
                skill_factor[N + i] = sf1
                skill_factor[N + i + 1] = sf1
            #elif sf1 != sf2 and np.random.rand() < rmp:
            #mfea-ii修改处
            elif sf1 != sf2 and np.random.rand() < rmp_matrix[sf1, sf2]:
                c1, c2 = util.x_two_point(p1, p2, sbxdi)
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
            else:
                p2  = util.find_relative(population, skill_factor, sf1, N)
                c1, c2 = util.x_two_point(p1, p2, sbxdi)
                c1 = mutate(query, c1)
                c2 = mutate(query, c2)
                c1, c2 = variable_swap(c1, c2, pswap)
                skill_factor[N + i] = sf1
                skill_factor[N + i + 1] = sf1

            population[N + i, :], population[N + i + 1, :] = c1[:], c2[:]
        
        # evaluate
        for i in range(N, 2 * N):
            sf = skill_factor[i]
            factorial_cost[i, sf] = functions[sf](query, population[i])
        scalar_fitness = calculate_scalar_fitness(factorial_cost)

        # sort
        sort_index = np.argsort(scalar_fitness)[::-1]
        population = population[sort_index]
        skill_factor = skill_factor[sort_index]
        factorial_cost = factorial_cost[sort_index]
        #mfea-ii修改处
        #scalar_fitness = scalar_fitness[sort_index]
        best_fitness = np.min(factorial_cost, axis=0)
        c1 = population[np.where(skill_factor == 0)][0]
        c2 = population[np.where(skill_factor == 1)][0]
        #mfea-ii修改处
        scalar_fitness = scalar_fitness[sort_index]

        # optimization info
        #message = {'algorithm': 'mfea', 'rmp':rmp}
        #mfea-ii修改处
        message = {'algorithm': 'mfea_ii', 'rmp':round(rmp_matrix[0, 1], 1)}
        results = get_optimization_results(t, population, factorial_cost, scalar_fitness, skill_factor, message)
        desc = 'gen:{} fitness:{} message:{}'.format(t, ' '.join('{:0.6f}'.format(res.fun) for res in results), message)
        ret = [res.fun for res in results]

        # statistic info
        gen_idx = query.cur_gen_idx
        first_phen = results[0].x
        first_val = results[0].fun
        #gen_idx, first_val, first_phen = self.select_offspring()
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
        
    return query.result
