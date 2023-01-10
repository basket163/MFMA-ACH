import numpy as np
import copy
import time
import itertools as it

import pkgMethod
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import edge
import util

import networkx as nx
import geatpy as ea

from globalVar import enumVar as enumVar

class MultiProblem(ea.Problem): # 继承Problem父类
    def __init__(self,scene):
        self.scene = scene
        userCount = globalVar.get_value(enumVar.userCount)
        
        lower = globalVar.get_value(enumVar.cellIdMin)
        upper = globalVar.get_value(enumVar.cellIdMax)

        #print("{}, {}".format(lower,upper,userCount))
        #input()

        name = 'MultiProblem' # 初始化name（函数名称，可以随意设置）
        M = 2 # 初始化M（目标维数）
        maxormins = [1, 1]#[1] * M # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = userCount # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [lower] * Dim # 决策变量下界
        ub = [upper] * Dim # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def aimFunc(self, pop): # 目标函数
        Vars = pop.Phen # 得到决策变量矩阵
        if 0 in Vars:
            #print("--Phen Vars has 0!!!")
            #print(Vars)
            lower = globalVar.get_value(enumVar.cellIdMin)
            Vars[Vars<lower]=lower
            #print("deal!")
            if 0 in Vars:
                print("still has 0")
                input()

        col_f1,col_f2 = computePhens(Vars,self.scene)

        evalution = np.hstack((col_f1,col_f2))

        pop.ObjV = evalution
        
        extraFitness = 0
        return extraFitness


def computePhens(Vars,scene):
    #scene = globalVar.get_value(enumVar.currentScene)
    popSize = globalVar.get_value(enumVar.nowPopSize)
    userCount = globalVar.get_value(enumVar.userCount)

    col_f1 = np.zeros((popSize,1),dtype=np.float32)
    col_f2 = np.zeros((popSize,1),dtype=np.float32)

    for i in range(popSize):
        arg_vec = Vars[i,:]
        tmpScene = copy.deepcopy(scene)
        newScene = edge.updateCurrentScene(tmpScene,arg_vec)


        cmni,cmpt,miga = edge.computeScene(newScene)
        engy = edge.computeEnergy(newScene)

        f1 = edge.statisticLatency(cmni,cmpt,miga)
        f2 = engy
        col_f1[i] = f1
        col_f2[i] = f2
    
    return col_f1,col_f2

def runMobjMain(scene, index):
    popSize = globalVar.get_value(enumVar.nowPopSize)
    maxGen = globalVar.get_value(enumVar.nowMaxGen)
    maxFitness = globalVar.get_value(enumVar.nowMaxFitness)
    p_c = globalVar.get_value(enumVar.nowpc)
    p_m = globalVar.get_value(enumVar.nowpm)
    p_l = globalVar.get_value(enumVar.nowpl)

    """===============================实例化问题对象==========================="""
    problem = MultiProblem(scene) # 生成问题对象
    """=================================种群设置==============================="""
    Encoding = 'RI'       # 编码方式
    NIND = popSize            # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """===============================算法参数设置============================="""
    myAlgorithm = ea.moea_NSGA2_templet(problem, population) # 实例化一个算法模板对象
    #print(f'Pc:{globalVar.pc} Pm:{globalVar.pm} maxGen:{globalVar.maxGen} maxFit:{globalVar.maxFit}')
    myAlgorithm.recOper.XOVR = p_c  #0.9 # 修改交叉算子的交叉概率
    myAlgorithm.mutOper.Pm = p_m  #0.2 # 修改变异算子的变异概率
    myAlgorithm.MAXGEN = maxGen # 最大进化代数
    myAlgorithm.MAXEVALS = maxFitness # 最大进化评价数
    myAlgorithm.drawing = 0 # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化======================="""
    NDSet = myAlgorithm.run()
    NDSet.save()
    #print('用时：%s 秒'%(myAlgorithm.passTime))
    #print('非支配个体数：%s 个'%(NDSet.sizes))
    #print('单位时间找到帕累托前沿点个数：%s 个'%(int(NDSet.sizes // myAlgorithm.passTime)))
    #col = 0
    #best_gen = np.argmin(problem.maxormins[col] * NDSet.ObjV[:, col]) # 记录最优种群个体是在哪一代


    #print('\n------a scene result:')
    #print('最优的目标函数值为：%s'%(best_ObjV))
    #print('最优的控制变量值为：\n{}'.format(best_Phen))
    #print('索引值是第 %s 代'%(best_gen + 1))
    #print('评价次数：%s'%(myAlgorithm.evalsNum))
    #print('时间已过 %s 秒'%(myAlgorithm.passTime))

    col_f1 = np.reshape(NDSet.ObjV[:,0], (-1,1))
    col_f2 = np.reshape(NDSet.ObjV[:,1], (-1,1))
    evalution = np.hstack((col_f1,col_f2))

    fileName = globalVar.get_value(enumVar.nowEvoPara)+"_mobj_"+str(index)

    edge.saveTimeCsv(evalution,fileName)

    bestLatencyIdxList = np.where(col_f1==np.min(col_f1))[0]
    bestLatencyIdx = bestLatencyIdxList[0]
    bestLatency = col_f1[bestLatencyIdx]
    bestLatencyEnergy = col_f2[bestLatencyIdx]
    bestPhen = NDSet.Phen[bestLatencyIdx,:]

    util.modifyFolder("Result","ResultMObj",index)

    #[population, obj_trace, var_trace] = myAlgorithm.run() # 执行算法模板
    #population.save() # 把最后一代种群的信息保存到文件中
    # 输出结果
    #best_gen = np.argmin(problem.maxormins * obj_trace[:, 1]) # 记录最优种群个体是在哪一代
    #best_ObjV = obj_trace[best_gen, 1]
    #best_Phen = var_trace[best_gen,:]

    '''print('\n------a scene result:')
    print('最优的目标函数值为：%s'%(best_ObjV))
    print('最优的控制变量值为：')
    for i in range(var_trace.shape[1]):
        print(var_trace[best_gen, i])
    print('有效进化代数：%s'%(obj_trace.shape[0]))
    print('最优的一代是第 %s 代'%(best_gen + 1))
    print('评价次数：%s'%(myAlgorithm.evalsNum))
    print('时间已过 %s 秒'%(myAlgorithm.passTime))'''

    #avgValMobjMain = bestLatency
    #avgEnyMobjMain = bestLatencyEnergy
    avgIterMobjMain = 1
    avgCallMobjMain = myAlgorithm.evalsNum

    return avgIterMobjMain,avgCallMobjMain,bestLatency,bestLatencyEnergy,bestPhen


def conductMObjMain():
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

        best_gen, best_call, best_ObjV, best_engy, best_Phen = runMobjMain(scene,index+1)

        best_Phen = np.array(best_Phen)
        if(len(best_Phen) == 0):
            print("gen {}, val {}, phen {} is null".format(best_gen,best_ObjV,best_Phen))
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