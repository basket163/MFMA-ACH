import numpy as np
import copy
import geatpy as ea

import pkgMethod
import globalVar
from globalVar import enumVar
import edge

class MultiProblem(ea.Problem): # 继承Problem父类
    def __init__(self,M = 2):
        userCount = globalVar.get_value(enumVar.userCount)
        lower = globalVar.get_value(enumVar.cellIdMin)
        upper = globalVar.get_value(enumVar.cellIdMax)

        name = 'MultiProblem' # 初始化name（函数名称，可以随意设置）
        #M = 2 # 初始化M（目标维数）
        Dim = userCount # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [lower] * Dim # 决策变量下界
        ub = [upper] * Dim # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def aimFunc(self, pop): # 目标函数
        Vars = pop.Phen # 得到决策变量矩阵
        evalution = aimMultiOP(Vars)
        '''print("--Phen Vars")
        print(Vars)
        print("evalution")
        print(evalution)'''
        pop.ObjV = evalution
        #pop.ObjV = np.hstack([f1, f2]) # 把求得的目标函数值赋值给种群pop的ObjV
        #pop.ObjV = x * np.sin(10 * np.pi * x) + 2.0 # 计算目标函数值，赋值给pop种群对象的ObjV属性

def conductSceneMultiOP():
    listScene_ori = globalVar.get_value(enumVar.listScene)
    listScene = copy.deepcopy(listScene_ori)
    index = 0
    for scene in listScene:
        index+=1
        globalVar.set_value(enumVar.currentScene,scene)
        print("\nmulti-----  scene{}:".format(index))
        print(scene)

        multiOP()
    return

def multiOP(G, listInit):
    popSize = globalVar.get_value(enumVar.nowPopSize)
    maxGen = globalVar.get_value(enumVar.nowMaxGen)
    maxFitness = globalVar.get_value(enumVar.nowMaxFitness)

    userCount = globalVar.get_value(enumVar.userCount)
    lower = globalVar.get_value(enumVar.cellIdMin)
    upper = globalVar.get_value(enumVar.cellIdMax)

    """================================实例化问题对象==========================="""
    problem = MultiProblem()       # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = 'RI'             # 编码方式
    NIND = popSize                   # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.moea_NSGA2_templet(problem, population) # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = maxGen    # 最大进化代数
    myAlgorithm.drawing = 0 # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化========================"""
    NDSet = myAlgorithm.run()   # 执行算法模板，得到帕累托最优解集NDSet
    NDSet.save()                # 把结果保存到文件中
    # 输出
    print('用时：%s 秒'%(myAlgorithm.passTime))
    print('非支配个体数：%s 个'%(NDSet.sizes))
    print('单位时间找到帕累托前沿点个数：%s 个'%(int(NDSet.sizes // myAlgorithm.passTime)))

    

def aimMultiOP(Phen): # define the aim function
    '''
    print("-----aim")
    print("--Phen")
    print(Phen)
    print("---LegV")
    print(LegV)
    input()
    '''
    
    f1,f2 = combineSolutions_Multi(Phen)
    evalution = np.hstack((f1,f2))
    #print("evalution:")
    #print(evalution)
    #input()
    return evalution
    #return [np.array([evalution]).T, LegV]

def combineSolutions_Multi(phenSolutions):
    scene = globalVar.get_value(enumVar.currentScene)
    popSize = globalVar.get_value(enumVar.nowPopSize)
    userCount = globalVar.get_value(enumVar.userCount)
    listSolution = []
    col_f1 = np.zeros((popSize,1),dtype=np.float32)
    col_f2 = np.zeros((popSize,1),dtype=np.float32)
    for i in range(popSize):
        #recombine = phenSolutions[i,:].reshape(userCount,1)
        newScene = copy.deepcopy(scene)
        newScene[12,:] = phenSolutions[i,:]
        f1,f2,f3 = edge.computeScene(newScene)
        #fitVal = f1
        col_f1[i]=f1
        col_f2[i]=f2
        #listSolution.append(fitVal)
    #print("listSolution")
    #print(listSolution)
    #print(colSolution)
    #input()
    return col_f1,col_f2