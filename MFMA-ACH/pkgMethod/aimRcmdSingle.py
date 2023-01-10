import numpy as np
import copy
import geatpy as ea

import pkgMethod
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import edge

class SingleProblemRcmd(ea.Problem): # 继承Problem父类
    def __init__(self,scene):
        userCount = globalVar.get_value(enumVar.userCount)
        lower = globalVar.get_value(enumVar.cellIdMin)
        upper = globalVar.get_value(enumVar.cellIdMax)
        #print("{}, {}".format(lower,upper))
        #input()

        name = 'SingleProblem' # 初始化name（函数名称，可以随意设置）
        M = 1 # 初始化M（目标维数）
        maxormins = [1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = userCount # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [lower] * Dim # 决策变量下界
        ub = [upper] * Dim # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        #自定义变量
        self.S = scene
    
    def aimFunc(self, pop): # 目标函数
        lower = globalVar.get_value(enumVar.cellIdMin)
        upper = globalVar.get_value(enumVar.cellIdMax)
        userCount = globalVar.get_value(enumVar.userCount) #edge.cfg.userCount
        Vars = pop.Phen # 得到决策变量矩阵
        if 0 in Vars:
            #print("--Phen Vars has 0!!!")
            #print(Vars)
            Vars[Vars<lower]=lower
            #print("deal!")
            if 0 in Vars:
                print("still has 0")
                input()
        #print("Vars\n{}".format(Vars))

        #community recommend
        #Vars[1,:] =  np.array([2, 1, 5, 2, 7, 4, 6, 10, 6, 3])
        #Vars[2,:] =  np.array([1, 1, 6, 6, 11, 4, 9, 13, 2, 7])
        #Vars[3,:] =  np.array([2, 1, 10, 5, 14, 4, 5, 16, 1, 7])
        #Vars[4,:] =  np.array([1, 1, 14, 2, 18, 4, 6, 15, 5, 7])
        pInServ = self.S[edge.cfg._pInServ,:]
        #Vars[0,:] =  np.array([2, 1, 5, 2, 7, 4, 6, 10, 6, 3])
        Vars[0,:] =  pInServ


        '''
        list1 = [2,4,6]
        listCondition = []
        x1 = Vars[:,[0]]
        x2 = Vars[:,[1]]
        x3 = Vars[:,[2]]
        x4 = Vars[:,[3]]
        '''
        fmlList = []
        for idx in np.arange(userCount):
            xi = Vars[:,[idx]]
            #find user xi current serv
            a_pInServ = self.S[edge.cfg._pInServ,idx]
            candiList = edge.getServListInHop(edge.cfg.lyapHop, a_pInServ)
            fmlCount = len(candiList)
            #fmlList = np.arange(fmlCount)
            fmlOne = 1
            for idxCdi in np.arange(fmlCount):
                cdi = candiList[idxCdi]
                fml = np.abs(xi-cdi)
                #fmlList[idxCdi] = fml
                fmlOne = fmlOne*fml
            #combine pop.CV
            #print("fmlOne\n{}".format(fmlOne))
            fmlList.append(fmlOne)
        #print("fmlList\n{}".format(fmlList))
        

        #print("x1\n{}".format(x1))
        #input()
        evalution = aimSingleOP(Vars)
        #print("evalution")
        #print(evalution)
        pop.ObjV = evalution
        #pop.ObjV = x * np.sin(10 * np.pi * x) + 2.0 # 计算目标函数值，赋值给pop种群对象的ObjV属性
        '''testCondition = [
            np.abs(x1-1)*np.abs(x1-2)*np.abs(x1-3),
            np.abs(x2-4)*np.abs(x2-5)*np.abs(x2-6),
            np.abs(x3-7)*np.abs(x3-8)*np.abs(x3-9),
            np.abs(x4-7)*np.abs(x4-8)*np.abs(x4-9)
        ]'''
        #print("testCondition\n{}".format(testCondition))
        #print("fmlArray\n{}".format(fmlArray))
        #input()
        '''pop.CV = np.hstack()'''
        #listCandi = [1,2,3]
        #exIdx1 = np.where(x1 not in listCandi)[0]
        pop.CV = np.hstack(fmlList)



def singleOPRcmd(scene):
    popSize = globalVar.get_value(enumVar.nowPopSize)
    maxGen = globalVar.get_value(enumVar.nowMaxGen)
    maxFitness = globalVar.get_value(enumVar.nowMaxFitness)

    """===============================实例化问题对象==========================="""
    problem = SingleProblemRcmd(scene) # 生成问题对象
    """=================================种群设置==============================="""
    Encoding = 'RI'       # 编码方式
    NIND = popSize            # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """===============================算法参数设置============================="""
    myAlgorithm = ea.soea_SEGA_templet(problem, population) # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = maxGen # 最大进化代数
    myAlgorithm.drawing = 0 # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化======================="""
    [population, obj_trace, var_trace] = myAlgorithm.run() # 执行算法模板
    #population.save() # 把最后一代种群的信息保存到文件中
    # 输出结果
    best_gen = np.argmin(problem.maxormins * obj_trace[:, 1]) # 记录最优种群个体是在哪一代
    best_ObjV = obj_trace[best_gen, 1]
    best_Phen = var_trace[best_gen,:]
    best_call = myAlgorithm.evalsNum
    best_Engy = edge.computeEnergyWithPhen(best_Phen)

    '''print('\n------a scene result:')
    print('最优的目标函数值为：%s'%(best_ObjV))
    print('最优的控制变量值为：')
    for i in range(var_trace.shape[1]):
        print(var_trace[best_gen, i])
    print('有效进化代数：%s'%(obj_trace.shape[0]))
    print('最优的一代是第 %s 代'%(best_gen + 1))
    print('评价次数：%s'%(myAlgorithm.evalsNum))
    print('时间已过 %s 秒'%(myAlgorithm.passTime))'''
    '''for num in var_trace[best_gen, :]:
        print(chr(int(num)), end = '')'''
    
    #opVal = best_ObjV,best_gen + 1,
    #print(best_Phen)
    
    return best_gen+1,best_call,best_ObjV,best_Engy,best_Phen

def aimSingleOP(Phen): # define the aim function
    
    '''print("-----aim")
    print("--Phen")
    print(Phen)'''
    #input()
    
    
    evalution = combineSolutions(Phen)
    #print("evalution:")
    #print(evalution)
    #input()
    return evalution
    #return [np.array([evalution]).T]

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


def conductRcmdSingleOP():
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

        best_gen, best_call, best_ObjV,best_Engy, best_Phen = singleOPRcmd(scene)
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