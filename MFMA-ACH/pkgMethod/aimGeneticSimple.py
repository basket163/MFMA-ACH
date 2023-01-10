import numpy as np
import copy

import pkgMethod
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import edge

def randomAresult(gene_N, lower, upper):
	phen = []
	#lower = 0
	#upper = int(gene_N/2)
	#print("lower is {} upper is {}".format(lower,upper))
	for i in range(gene_N):
		x = np.random.randint(lower, upper+1)
		phen.append(x)
	#print (phen)
	phen = np.array(phen)
	return phen

def randomAllresult(population_N,gene_N, lower, upper):
	phens = []
	for i in range(population_N):
		phens.append(randomAresult(gene_N, lower, upper))
	phens = np.array(phens)
	return phens

def combineSolutions(phenSolutions):
    scene = globalVar.get_value(enumVar.currentScene)
    popSize = globalVar.get_value(enumVar.nowPopSize)
    userCount = globalVar.get_value(enumVar.userCount)
    #listSolution = []
    colFitness = np.zeros((popSize,1),dtype=np.float32)
    '''if 0 in phenSolutions:
        print("0 in phenSolutions")
        input()'''
    for i in range(popSize):
        #recombine = phenSolutions[i,:].reshape(userCount,1)
        tmpScene = copy.deepcopy(scene)
        #newScene[12,:] = phenSolutions[i,:]
        if 0 in phenSolutions[i,:]:
            print("phen has 0!!!")
            print(i)
            print(phenSolutions[i,:])
            input()
        newScene = edge.updateCurrentScene(tmpScene,phenSolutions[i,:])
        cmni,cmpt,miga = edge.computeScene(newScene)
        #fitVal = cmni,cmpt,miga
        fitVal = edge.statisticLatency(cmni,cmpt,miga)
        colFitness[i]=fitVal
        #listSolution.append(fitVal)
    #print("listSolution")
    #print(listSolution)
    #print(colFitness)
    #input()
    return colFitness



GA_ELITRATE= 0.10 # elitism rate
GA_MUTATIONRATE = 0.25 # mutation rate

def mate(population, pop, lower, upper, GA_POPSIZE, GA_TARGET_len):
	esize = int(GA_POPSIZE * GA_ELITRATE)
	'''
	print("1")
	printpop(population)
	print()
	printpop(pop)
	'''
	for i in range(esize): # Elitism
		pop[i] = population[i]
	for i in range(esize, GA_POPSIZE):
		i1 = np.random.randint(0, GA_POPSIZE / 2)
		i2 = np.random.randint(0, GA_POPSIZE / 2)
		spos = np.random.randint(0, GA_TARGET_len)
		#print("i1 {},i2 {},spos {}".format(i1,i2,spos))
		#print(population[i1])
		#print(population[i2])
		#print(population[i1][:spos])
		#print(population[i2][spos:])
		#input()
		#pop[i] = population[i1][:spos] + population[i2][spos:] # Mate
		pop[i] = np.concatenate([population[i1][:spos], population[i2][spos:]])
		#print(pop[i])
		#input()
		if np.random.random() < GA_MUTATIONRATE: # Mutate
			pos = np.random.randint(0, GA_TARGET_len -1)
			#pop[i] = pop[i][:pos] + random.randint(lower, upper) + pop[i][pos+1:]
			pop[i][pos] = np.random.randint(lower, upper)
	'''
	print("\n2")
	printpop(population)
	print()
	printpop(pop)
	print()
	input()
	'''

def printpop(pp):
	for p in pp:
		print(p)


def runGeneticSimple(scene):
	#print("\n=======Running GeneticSimple ...")

	userCount = globalVar.get_value(enumVar.userCount)
	popSize = globalVar.get_value(enumVar.nowPopSize)
	maxGen = globalVar.get_value(enumVar.nowMaxGen)
	maxFitness = globalVar.get_value(enumVar.nowMaxFitness)

	# node range list

	#lower = 1 
	#upper = 3 #2 ** bit_num
	#lower = method.daGetLower()   # 1 
	#upper = method.daGetUpper()   # 3
	#numNode = method.daGetNodesNum()
	lower = globalVar.get_value(enumVar.cellIdMin)
	upper = globalVar.get_value(enumVar.cellIdMax)


	#GA_POPSIZE = 10
	GA_TARGET_len = userCount
	#max_epochs = 5
	#GA_POPSIZE = method.daGetPopSize() #10
	GA_POPSIZE = popSize

	#generate init value
	population = randomAllresult(GA_POPSIZE,GA_TARGET_len,lower, upper)
	pop	 = randomAllresult(GA_POPSIZE,GA_TARGET_len,lower, upper)
	#print(population.shape)
	#print(population)
	#input()

	#return values
	indexGen = 0
	indexCall = 0
	bestGen = 0
	bestCall = 0
	bestVal = 1000
	bestPhen = []

	while True:
		if (indexGen > maxGen):
			print("reach maxGen {} / {}".format(indexGen, maxGen))
			break
		if (indexCall > maxFitness):
			print("reach maxFitness {} / {}".format(indexCall, maxFitness))
			break

		indexGen+=1
		
		#population = np.reshape(GA_POPSIZE,GA_TARGET_len)
		colFitness = combineSolutions(population)
		#colFitness = np.reshape(GA_POPSIZE,1)
		#print(colFitness)
		#input()
		popFit = np.hstack((colFitness,population))
		
		
		popFit = sorted(popFit, key=lambda c: c[0])
		#print(population)
		#print(popFit)
		#input()
		firstP =  popFit[0][0]
		bestVal = firstP
		bestPhen = population[0]
		bestGen = indexGen
		bestCall = indexCall
		#print(firstP)
		#input()

		indexCall += GA_POPSIZE

			
		if(firstP < bestVal):
			bestVal = firstP
			bestPhen = population[0]
			bestGen = indexGen
			bestCall = indexCall
			#if globalVar.get_value(enumVar.show) == 1:
			#print("simple genetic current: {} / {}".format(indexGen, maxGen))
			#print("best: {}, current： {} = {}".format(bestVal,firstP,population[0]))
		popFit = np.array(popFit)
		#print(popFit.shape)
		#input()
		population = popFit[:,1:]
		#print(population)
		#input()
		#print(pop)
		#input()

		mate(population, pop, lower, upper, GA_POPSIZE, GA_TARGET_len)
		population, pop = pop, population

	bestEngy = edge.computeEnergyWithPhen(bestPhen)

	return bestGen, bestCall, bestVal, bestEngy, bestPhen

def conductGeneticSimple():
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
		scene = listScene[index]
		globalVar.set_value(enumVar.currentScene,scene)

		best_gen,best_call,best_ObjV,best_Engy,best_Phen = runGeneticSimple(scene)
		#print("gen {}, val {}, phen {}".format(best_gen,best_ObjV,best_Phen))
		#input()
		### important update
		scene = edge.updateCurrentScene(scene,best_Phen)

		#valOP = [best_ObjV, best_gen]
		
		valStack[1,index] = best_gen
		valStack[2,index] = best_call
		valStack[3,index] = best_ObjV
		valStack[4,index] = best_Engy
		phenStack = np.vstack((phenStack,best_Phen))

		#index+=1
		if index < lastIdx:
			nextScene = edge.updateNextScene(listScene,index)
			listScene[index+1] = nextScene
	return valStack,phenStack