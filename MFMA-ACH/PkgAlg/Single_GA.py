import numpy as np
import itertools as it
import copy

import PkgAlg
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import edge

class Individual(object):
    def __init__(self, evo_cls, phenotypes):
        self.phenotypes = phenotypes  # phenotype
        self.fitness = None
        self.fitness = compute_fitness(evo_cls, self.phenotypes)  # value of the fitness function
        evo_cls.indexCall += 1


    def __str__(self):
        return '{0} = {1}'.format(self.phenotypes, self.fitness)

class GA_single(object):
    """ This is evo_cls. """

    def __init__(self, query):
        self.query = query
        self.is_single = True
        
        self.result = query.result

        self.scene = query.scene
        self.user_count = query.user_count
        self.pop_size = query.pop_size
        self.max_gen = query.max_gen
        self.max_fit = query.max_fit
        self.pc = query.pc
        self.pm = query.pm
        self.pl = query.pl
        self.cell_id_min =  query.cell_id_min
        self.cell_id_max = query.cell_id_max
        #print(f'{scene}')
        #print(f'{userCount} {popSize} {maxGen} {maxFitness} {p_c} {p_m} {p_l} {lower} {upper}')
        #input()

        self.population = None
        self.fit_array = None
        self.parent = None
        self.offspring = None

        #return values
        self.cur_gen_idx = 0
        self.cur_fit_idx = 0

        '''
        self.best_gen_idx = 0
        self.best_gen_val = 0
        self.best_gen_phen = None
        self.best_fit_idx = 0
        self.best_fit_val = 0
        self.best_fit_phen = None
        '''

    def generate_population(self, row, col):
        low = self.cell_id_min
        high = self.cell_id_max+1
        rand_vals = np.random.randint(low, high, (row, col) )

        #self.population = [Individual(self, p) for p in rand_vals]
        #self.fit_col = np.array([x.fitness for x in self.population]).reshape(-1,1)
        #self.fit_col = np.array([ compute_fitness(self, p) for p in rand_vals ]).reshape(-1,1)
        self.fit_array = self.compute_fit_list(rand_vals)

        self.parent = copy.deepcopy(rand_vals)
        self.offspring = copy.deepcopy(rand_vals)

        #print([x.phenotypes for x in self.population])
        #print([x.fitness for x in self.population])

    def compute_fit_list(self, phens):
        ret = []
        for p in phens:
            self.cur_fit_idx += 1
            fit_val = compute_fitness(self, p)
            ret.append(fit_val)

            if(self.cur_fit_idx == 1):
                self.result.best_fit_idx = self.cur_fit_idx
                self.result.best_fit_val = fit_val
                self.result.best_fit_phen = p
                self.result.add_fit_result(self.result.best_fit_idx, self.result.best_fit_val, self.result.best_fit_phen)

            if(fit_val < self.result.best_fit_val):
                self.result.best_fit_idx = self.cur_fit_idx
                self.result.best_fit_val = fit_val
                self.result.best_fit_phen = p
                self.result.add_fit_result(self.result.best_fit_idx, self.result.best_fit_val, self.result.best_fit_phen)

            self.result.add_all_fit_result(self.cur_fit_idx, self.result.best_fit_val, self.result.best_fit_phen)
        ret_array = np.array(ret)
        return ret_array

    def crossover(self):
        if (self.pop_size & 1) != 0:
            print('in corssover, population number is not even.')
            input()
        half_pop_size = int(self.pop_size / 2)
        half_cross_num = int( half_pop_size * self.pc)
        odd_idx_list = np.arange(0, self.pop_size, 2)
        cross_chrom_idx = np.random.choice(odd_idx_list, size=half_cross_num, replace=False)
        for x in cross_chrom_idx:
            #print(f'{x}, {x+1}')
            r = np.random.random(1)
            if(r<self.pc):
                #print(f'{r}<{self.para.evoPc}')
                crossPoint = np.random.randint(0,self.user_count)
                #print(f'crossPoint {crossPoint}/{self.popCol}')
                lastPoint = self.user_count
                c1 = copy.deepcopy(self.parent[x,:])
                c2 = copy.deepcopy(self.parent[x+1,:])
                #print(f'c1\n{c1}')
                #print(f'c2\n{c2}')
                #print(f'crossPoint\n{crossPoint}/{len(c1)}')
                #print(c1[0:crossPoint])
                #print(c2[crossPoint:self.popCol+1])
                c3 = np.hstack((c1[0:crossPoint], c2[crossPoint:lastPoint] ))
                c4 = np.hstack((c2[0:crossPoint], c1[crossPoint:lastPoint] ))
                #print(f'c3\n{c3}')
                #print(f'c4\n{c4}')
                self.offspring[x,:] = c3
                self.offspring[x+1,:] = c4
        '''
        print(self.parent)
        print()
        print(self.offspring)
        input()
        '''
        return self.offspring

    def mutation(self):
        idx_list = np.arange(self.user_count)
        muta_num = int(self.user_count * self.pm)
        for x in np.arange(self.pop_size):
            # mutation_point = np.random.randint(0,self.user_count)
            muta_in_chrom_idx = np.random.choice(idx_list, size=muta_num, replace=False)
            for y in muta_in_chrom_idx:
                mutation_context = np.random.randint(self.cell_id_min, self.cell_id_max + 1)
                # print(f'mutationPoint row:{x} col:{mutation_point} -> {mutation_context}')
                self.offspring[x, y] = mutation_context
        return self.offspring


    def select_offspring(self):
        # save first phen and select other phens by rank probobility
        combine_pop = np.vstack((self.parent, self.offspring))
        combine_idx = np.arange(2*self.pop_size)
        #combine_fit = [ compute_fitness(self, p) for p in combine_pop ]
        combine_fit = self.compute_fit_list(combine_pop)

        #cc = np.hstack((combine_idx.reshape(-1,1), np.array(combine_fit).reshape(-1,1),combine_pop))
        #print('cc')
        #print(cc)

        first_idx = np.argmin(combine_fit)
        first_fit = combine_fit[first_idx]
        first_phen = combine_pop[first_idx]
        #print('--')
        #print(first_idx)

        # sort
        #index = np.argsort(-np.array(combine_fit)) #desc
        #rank = np.argsort(index)+1
        index_score = np.argsort(-np.array(combine_fit))
        rank_score = np.argsort(index_score)+1
        
        combine_size = 2 * self.pop_size
        all_score = self.pop_size*(combine_size+1)
        rank_probability = np.around(np.true_divide(rank_score, all_score), decimals=4)
        # rank_probability must be sum to 1
        selection = np.random.choice(combine_idx, size=self.pop_size, replace=False, p=rank_probability)

        '''
        # 取消科学计数法显示
        np.set_printoptions(suppress=True)
        v = np.hstack((
            combine_idx.reshape(-1,1),
            np.array(combine_fit).reshape(-1,1),
            rank_score.reshape(-1,1),
            rank_probability.reshape(-1,1)
            ))
        print(v)
        print(selection)
        '''

        # remain elitism
        if first_idx not in selection:
            selection[0] = first_idx
        #print(selection)
        

        for i, s in enumerate(selection):
            self.parent[i] = combine_pop[s]
            self.fit_array[i] = combine_fit[s]
        self.offspring = copy.deepcopy(self.parent)


        return self.cur_gen_idx, first_fit, first_phen
        

        


    def run(self, query):
        #reset
        self.cur_gen_idx = 0
        self.cur_fit_idx = 0

        self.generate_population(self.pop_size, self.user_count)


        while True:
            #for mfea
            query.cur_gen_idx += 1

            self.cur_gen_idx += 1

            if (self.cur_gen_idx > self.max_gen):
                #print("exceed max_gen: {0}/{1} ".format(self.cur_gen_idx, self.max_gen))
                break

            if self.cur_fit_idx > self.max_fit:
                #print("exceed max_fit: {0}/{1} ".format(self.cur_fit_idx, self.max_fit))
                break

            self.crossover()
            self.mutation()
            gen_idx, first_val, first_phen = self.select_offspring()
            

            if(self.cur_gen_idx == 1):
                self.result.best_gen_idx = gen_idx
                self.result.best_gen_val = first_val
                self.result.best_gen_phen = first_phen
                self.result.add_gen_result(self.result.best_gen_idx, 
                                            self.result.best_gen_val, 
                                            self.result.best_gen_phen)

            if(first_val < self.result.best_gen_val):
                self.result.best_gen_idx = gen_idx
                self.result.best_gen_val = first_val
                self.result.best_gen_phen = first_phen
                self.result.add_gen_result(self.result.best_gen_idx, 
                                            self.result.best_gen_val, 
                                            self.result.best_gen_phen)

            self.result.add_all_gen_result(self.cur_gen_idx, 
                                            self.result.best_gen_val, 
                                            self.result.best_gen_phen)

            #for mfea
            # statistic info
            gen_idx = query.cur_gen_idx
            first_phen = first_phen
            first_val = first_val
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
        return self.result

    def memetic(self):
        #self.population = [Individual(p,self) for p in rand_vals]
        #self.best = sorted(self.population, key=lambda x: x.fitness)[0]

        scene = globalVar.get_value(enumVar.currentScene)
        popSize = globalVar.get_value(enumVar.nowPopSize)
        userCount = globalVar.get_value(enumVar.userCount)

        for i in self.population:
            arg_vec = i.phenotypes

            tmpScene = copy.deepcopy(scene)

            newScene = edge.updateCurrentScene(tmpScene,arg_vec)
            cmni,cmpt,miga = edge.computeScene(newScene)
            #fitVal = cmni,cmpt,miga
            fitVal = edge.statisticLatency(cmni,cmpt,miga)

            new_vec = copy.deepcopy(arg_vec)

            #generate local search space
            dictUserServ = generateCommunity(newScene)

            #listCommunity = []
            for k in dictUserServ.keys():
                for v in dictUserServ[k]:
                    tmp_vec = copy.deepcopy(arg_vec)
                    tmpScene = copy.deepcopy(scene)
                    tmp_vec[k-1] = v
                    comScene = edge.updateCurrentScene(tmpScene,tmp_vec)
                    #listCommunity.append(comScene)
                    cmni,cmpt,miga = edge.computeScene(comScene)
                    comVal = edge.statisticLatency(cmni,cmpt,miga)
                    if comVal < fitVal:
                        fitVal = comVal
                        new_vec = copy.deepcopy(tmp_vec)

            i.phenotypes = new_vec
            i.fitness = fitVal
            #return new_vec, fitVal





def generateCommunity(scene):
    dictUserServ = {}
    userCount = globalVar.get_value(enumVar.userCount)
    user = scene[edge.cfg._userId,:]
    inCell = scene[edge.cfg._uInCell,:]

    for idx in np.arange(userCount):
        x = inCell[idx]
        #find user xi current serv
        candiList = edge.getServListInHop(1, x)
        dictUserServ[idx+1] = candiList
    return dictUserServ

def compute_fitness(evo_cls, arg_vec):
    config = evo_cls.query.config
    tmpScene = copy.deepcopy(evo_cls.scene)
    #evo_cls.cur_fit_idx += 1
    

    newScene = edge.updateCurrentScene(config, tmpScene,arg_vec)
    cmni,cmpt,miga = edge.computeScene(config, newScene)
    fitVal = edge.statisticLatency(cmni,cmpt,miga)

    #evo_cls.result.add_all_fit_result(evo_cls.cur_fit_idx, fitVal, arg_vec)
    
    #for mfea
    query = evo_cls.query
    query.cur_fit_idx += 1

    fit_val = fitVal
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


    if(query.cur_fit_idx == 1):
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


    return fitVal

    

def run_single_GA(query):
    #print("\n-----Running Gene algorithm ...")
    query.cur_fit_idx = 0
    query.cur_gen_idx = 0

    ga = GA_single(query)
    config = query.config

    # result = globalVar.Result()
    ga.run(query)

    '''
    print('change:')
    for x,y in ga.result.dict_gen.items():
        print(f"{x}: {y}")
    print('all:')
    for x,y in ga.result.dict_all_gen.items():
        print(f"{x}: {y}")
    
    print('change:')
    for x,y in ga.result.dict_fit.items():
        print(f"{x}: {y}")
    print('all:')
    for x,y in ga.result.dict_all_fit.items():
        print(f"{x}: {y}")

    print(f'evo \nbest_gen_idx: {ga.best_gen_idx}, best_gen_val: {ga.best_gen_val}')
    print(f'best_fit_idx: {ga.best_fit_idx}, best_fit_val: {ga.best_fit_val}')
    '''

    #print(f'evo result: best_gen_idx {ga.result.best_gen_idx}, best_fit {ga.result.best_fit_idx}, best_val {ga.result.best_val}')
    #print(f'best_phen: {ga.result.best_phen}\n')


    bestEngy = edge.computeEnergyWithPhen(config, ga.result.best_phen)
    #arrayGen = np.array(ga.listGen).reshape((-1,1))
    #arrayGenBest = np.array(ga.listGenBest).reshape((-1,1))
    #table = np.hstack((arrayGen,arrayGenBest))

    table = ga.result.get_dict_as_table(ga.result.dict_gen)
    fileName = ga.query.get_file_name()
    title = ['gen_idx', 'best_val']
    edge.save_table_to_csv(config, table, title, fileName,
                            prefix = globalVar.log_str_detail, 
                            surfix = globalVar.log_str_gen)

    table2 = ga.result.get_dict_as_table(ga.result.dict_fit)
    fileName2 = ga.query.get_file_name()
    title2 = ['fit_idx', 'best_val']
    edge.save_table_to_csv(config, table2, title2, fileName2, prefix='log', surfix='fit')

    return query

def conduct_GA_single(query):
    query.cur_fit_idx = 0
    query.cur_gen_idx = 0
    result = query.result
    
    listScene_ori = globalVar.get_value(enumVar.listScene)
    listScene = copy.deepcopy(listScene_ori)
    userCount = globalVar.get_value(enumVar.userCount)
    #index = 0
    slotNum = len(listScene)
    lastIdx = slotNum-1

    valStack = np.zeros((5,slotNum))
    valStack[0,:] = np.arange(1,slotNum+1)
    phenStack = np.arange(1,userCount+1,1)
    for index in range(slotNum):
        nowPara = globalVar.get_value(enumVar.nowEvoPara)
        print("para: {}, slot index: {}".format(nowPara,index+1))

        scene = listScene[index]
        globalVar.set_value(enumVar.currentScene,scene)

        best_gen, best_call, best_ObjV, best_engy, best_Phen = runMemeMain(scene,query)

        best_Phen = np.array(best_Phen)
        if(len(best_Phen) == 0):
            print("gen {}, val {}, phen {}".format(best_gen,best_ObjV,best_Phen))
            input()
        
        ### important update
        scene = edge.updateCurrentScene(scene,best_Phen)

        #valOP = [best_ObjV, best_gen]
        
        valStack[1,index] = best_gen
        valStack[2,index] = best_call
        valStack[3,index] = best_ObjV
        valStack[4,index] = best_engy
        phenStack = np.vstack((phenStack,best_Phen))

        #index+=1
        if index < lastIdx:
            nextScene = edge.updateNextScene(listScene,index)
            listScene[index+1] = nextScene
    return valStack,phenStack