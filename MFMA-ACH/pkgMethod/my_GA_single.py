import numpy as np
import itertools as it
import copy

import pkgMethod
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import edge

class GA(object):

    def __init__(self,
                query,
                l_bound,
                u_bound,
                prob_cross,
                prob_mut,
                prob_local,
                cross, #reduced_surrogate
                mutation, #single_point
                selection_parents, #parent.inbreeding_h
                selection_offspring, #offspring.elite
                population=None
                ):
        self.l_bound = l_bound
        self.h_bound = u_bound
        self.population = population
        self.pop_fit = None
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut
        self.prob_local = prob_local
        self.f_cross = cross
        self.f_mutation = mutation
        self.f_selector_parents = selection_parents
        self.f_selector_offspring = selection_offspring
        self.parents = None
        self.offspring = None
        self.best = None
        self.query = query

        #return values
        self.indexGen = 0
        self.indexCall = 0
        self.bestGen = 0
        self.bestCall = 0
        self.bestVal = 1000
        self.bestPhen = []
        self.listGen = []
        self.listGenBest = []

    def generate_population(self, size, fitness_vector_size):
        #print(size)
        #print(fitness_vector_size)
        rand_vals = np.random.randint(self.l_bound,
                                       self.h_bound,
                                       (size, fitness_vector_size))
        ###不增加pagerank初始化
        '''
        for i, x in enumerate(rand_vals):
            ii = i%len(listInit)
            rand_vals[i] = np.array(listInit[ii])
        '''
        #print(rand_vals)
        self.population = [Individual(p,self) for p in rand_vals]
        self.pop_fit = [x.fitness for x in self.population]

        #self.best = sorted(self.population, key=lambda x: x.fitness)[0]
        #print("init done.")
        #print(self.best)

    def select_parents(self):
        self.parents = self.f_selector_parents(self.population)
        #print("in select_parents")
        '''for p in self.parents:
            for q in p:
                print(q.phenotypes)
        print("end")
        input()'''
        #print("self.parents")
        #printpop(self.parents)

    def perform_crossover(self):
        self.offspring = list(it.chain.from_iterable([self.f_cross(*x, self.prob_cross) for x in self.parents]))
        #printpop(self.offspring)


    def mutate(self):
        self.offspring = [Individual(self.f_mutation(x, self.prob_mut, self.l_bound, self.h_bound),self) for x in self.offspring]
        #printpop(self.offspring)
        #print("pass mutate")

    def select_offspring(self):
        self.offspring = self.f_selector_offspring(par=self.parents, offspr=self.offspring)
        #printpop(self.offspring)
        #print("pass select_offspring")

    def set_population(self, new_popul=None):
        self.population = self.offspring if new_popul is None else new_popul
        #print("pass set_population")

    def run(self, population_size, ff_vec_size, max_epochs, max_fitness):
        self.generate_population(population_size, ff_vec_size)
        #print('Initial population')
        #for ind in sorted(ga.population, key=lambda x: x.fitness):
        #   print(ind, ind.chromosomes())
        #for i in range(max_epochs):
        self.indexGen = 0
        self.indexCall = 0
        self.bestGen = 0
        self.bestCall = 0
        self.listGen.clear()
        self.listGenBest.clear()

        while True:
            #reach = max_epochs  #get_value(enumVar.reachMaxFitness)
            if (self.indexGen >= max_epochs):
                print("reach maxGen, Meme: {0}/{1} ".format(self.indexGen + 1, max_epochs))
                break

            self.indexGen+=1

            if self.indexCall >= max_fitness:
                print("reach max_fitness, Meme: {0}/{1} ".format(self.indexCall + 1, max_fitness))
                break
            #print("before---")
            #print(p for p in self.population)
            self.select_parents()
            self.perform_crossover()
            self.mutate()
            self.memetic()
            self.select_offspring()
            self.set_population()

            
            b = sorted(self.population, key=lambda x: x.fitness)[0]
            #self.bestGen = self.indexGen
            #self.bestCall = self.indexCall
            self.bestVal = b.fitness
            self.bestPhen = b.phenotypes


            
            if b.fitness < self.best.fitness:
                self.best = b
                #if globalVar.get_value(enumVar.show) == 1:
                #print('Gene: {0}/{1} Current population:'.format(self.indexGen + 1, max_epochs))
                #print("{}".format(b))

                self.bestGen = self.indexGen
                self.bestCall = self.indexCall
                self.bestVal = b.fitness
                self.bestPhen = b.phenotypes
            #else:
            #   self.best
            self.listGen.append(self.indexGen)
            self.listGenBest.append(self.bestVal)

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


class Individual(object):
    def __init__(self, phenotypes, ga):
        self.phenotypes = phenotypes  # phenotype
        self.fitness = fitness_funcLocal(self.phenotypes)  # value of the fitness function
        #after memetic
        #self.phenotypes = vec  # phenotype
        ga.indexCall += 1

    def chromosomes(self):
        chrom = []
        for phenotype in self.phenotypes:
            if phenotype < 0:
                num = '-{0}' + str(bin(phenotype))[3:]
            else:
                num = '+{0}' + str(bin(phenotype))[2:]
            chrom.append(num.format((bit_num - len(num[4:])) * '0'))
        return chrom

    def __str__(self):
        #return '{0} = {1}'.format(real(self.phenotypes), self.fitness)
        return '{0} = {1}'.format(self.phenotypes, self.fitness)

def real(x):
    return [y * eps + interval[0] for y in x]

def printpop(pops):
    print("---")
    for pop in pops:
        for p in pop:
            print(p)
    print("---")

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

def fitness_funcLocal(arg_vec):

    scene = globalVar.get_value(enumVar.currentScene)
    popSize = globalVar.get_value(enumVar.nowPopSize)
    userCount = globalVar.get_value(enumVar.userCount)

    tmpScene = copy.deepcopy(scene)

    newScene = edge.updateCurrentScene(tmpScene,arg_vec)
    cmni,cmpt,miga = edge.computeScene(newScene)
    #fitVal = cmni,cmpt,miga
    fitVal = edge.statisticLatency(cmni,cmpt,miga)

    return fitVal

    

def runMemeMain(scene,query):
    #print("\n-----Running Gene algorithm ...")
    userCount = globalVar.get_value(enumVar.userCount)
    popSize = globalVar.get_value(enumVar.nowPopSize)
    maxGen = globalVar.get_value(enumVar.nowMaxGen)
    maxFitness = globalVar.get_value(enumVar.nowMaxFitness)
    p_c = globalVar.get_value(enumVar.nowpc)
    p_m = globalVar.get_value(enumVar.nowpm)
    p_l = globalVar.get_value(enumVar.nowpl)
    lower = globalVar.get_value(enumVar.cellIdMin)
    upper = globalVar.get_value(enumVar.cellIdMax)

    ga = GA(query,
            lower,
            upper,
            p_c,  # crossover prob
            p_m,  # mutation prob
            p_l,
            pkgMethod.reduced_surrogate, #crossover
            pkgMethod.single_point,  #mutator
            pkgMethod.parent.inbreeding_h,  #selector
            pkgMethod.offspring.elite  #selector
            )

    ga.run(popSize, userCount, maxGen, maxFitness)

    thebest = ga.best
    #print("{} = {}".format(thebest.phenotypes, thebest.fitness))
    
    bestGen = ga.bestGen+1
    bestCall = ga.bestCall
    bestVal = ga.bestVal
    bestPhen = ga.bestPhen
    bestEngy = edge.computeEnergyWithPhen(bestPhen)
    #print("{} = {}".format(bestPhen, bestVal))
    #print("gen {}, call {}".format(bestGen, bestCall))
    #input()


    arrayGen = np.array(ga.listGen).reshape((-1,1))
    arrayGenBest = np.array(ga.listGenBest).reshape((-1,1))
    table = np.hstack((arrayGen,arrayGenBest))
    fileName = globalVar.get_value(enumVar.nowEvoPara)
    edge.saveTimeCsv(table,fileName)
    

    return bestGen, bestCall, bestVal, bestEngy, bestPhen

def conduct_my_GA_single(query):
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