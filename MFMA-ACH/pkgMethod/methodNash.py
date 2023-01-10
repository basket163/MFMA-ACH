import numpy as np

import pkgMethod
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import edge



def nash(scene):
    #every user find a best profile placement
    """scene: 0 userId, 1 uInCell,  * 2 userRequ, 3 inServPflNum, 4 inServCapa,
    5 uInCellLast, * 6 userRequLast, 7 inServPflNumLast, 8 inServCapaLast, 9 pInServLast,
    10 userQueuLast, 11 userQueu, 12 pInServ (solution)"""
    '''
    _userId = globalVar.get_value(enumScene._userId)
    _uInCell = globalVar.get_value(enumScene._uInCell)
    _userRequ = globalVar.get_value(enumScene._userRequ)
    _inServPflNum = globalVar.get_value(enumScene._inServPflNum)
    _inServCapa = globalVar.get_value(enumScene._inServCapa)
    _uInCellLast = globalVar.get_value(enumScene._uInCellLast)
    _userRequLast = globalVar.get_value(enumScene._userRequLast)
    _inServPflNumLast = globalVar.get_value(enumScene._inServPflNumLast)
    _inServCapaLast = globalVar.get_value(enumScene._inServCapaLast)
    _pInServLast = globalVar.get_value(enumScene._pInServLast)
    _userQueuLast = globalVar.get_value(enumScene._userQueuLast)
    _userQueu = globalVar.get_value(enumScene._userQueu)
    _pInServ = globalVar.get_value(enumScene._pInServ)
    '''
    cfg = edge.cfg
    userCount = globalVar.get_value(enumVar.userCount)
    #for u in scene[cfg._userId,:]:
    #    print(u)
    # 1) find servs less than hop
    #servs = scene[cfg._pInServ,:]
    #print("in a nash, for scene:{}".format(scene))
    nashServs = []
    for userIdx in np.arange(userCount):
        #print("**userIdx: {}".format(userIdx))
        s = scene[cfg._pInServ,userIdx]
        servLessThanHop = edge.getServListInHop(cfg.nashHopIn,s)
        #print("serv {} in hop: {}".format(s,servLessThanHop))
        # 2) find a best serv
        bestServs = edge.findBestServForUser(s,userIdx,servLessThanHop,scene)
        #because bestServs exists multi values, so random select a best value
        #print("serv: {}, candidate: {}".format(s,bestServs))
        #input()
        best = np.random.choice(bestServs)
        #best2 = np.random.choice(bestServs,size=1)
        #print("best {}, best2 {}".format(best,best2))
        #input()
        nashServs.append(best)
    # 3) update
    nashServ = np.array(nashServs)
    nashScene = edge.updateCurrentScene(scene,nashServ)

    return nashScene

def recursiveNash(n,scene):
    if n <= 0:
        #print("n == 0, return")
        return scene
    else:
        nash_scene = nash(scene)
        #print("n {}, nash_scene:\n{}".format(n,nash_scene))
        n = n - 1
        return recursiveNash(n,nash_scene)

def conductNash(listScene):
    nashCount = edge.cfg.nashRunCount
    #sys.setrecursionlimit(nashCount*10)
    userCount = globalVar.get_value(enumVar.userCount)

    slotNum = len(listScene)
    lastIdx = slotNum-1
    _pInServ = globalVar.get_value(enumScene._pInServ)
    valStack = np.zeros((5,slotNum))
    valStack[0,:] = np.arange(1,slotNum+1)

    phenStack = np.arange(1,userCount+1,1)
    for index in range(slotNum):  # scene in listScene:
        scene = listScene[index]
        globalVar.set_value(enumVar.currentScene,scene)
        
        lastScene = recursiveNash(nashCount,scene)
        #print("lastScene")
        #print(lastScene)
        cmni,cmpt,miga = edge.computeScene(lastScene)
        best_ObjV = edge.statisticLatency(cmni,cmpt,miga)
        engy = edge.computeEnergy(lastScene)
        best_Phen = lastScene[_pInServ,:]
        valStack[1,index] = 1
        valStack[2,index] = 1
        valStack[3,index] = best_ObjV
        valStack[4,index] = engy
        phenStack = np.vstack((phenStack,best_Phen))

        if index < lastIdx:
            nextScene = edge.updateNextScene(listScene,index)
            listScene[index+1] = nextScene

    return valStack,phenStack
