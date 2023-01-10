import numpy as np

import pkgMethod
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import edge

def conductFollow(listScene):
    """Instant follow"""
    userCount = globalVar.get_value(enumVar.userCount)

    slotNum = len(listScene)
    lastIdx = slotNum-1
    _pInServ = globalVar.get_value(enumScene._pInServ)
    valStack = np.zeros((5,slotNum)) # row 0 is slot index, row 1 is latency, row 2 is energy
    valStack[0,:] = np.arange(1,slotNum+1)

    phenStack = np.arange(1,userCount+1,1)
    for index in range(slotNum):  # scene in listScene:
        scene = listScene[index]
        globalVar.set_value(enumVar.currentScene,scene)

        best_Phen = scene[_pInServ,:]
        ### important update
        scene = edge.updateCurrentScene(scene,best_Phen)

        cmni,cmpt,miga = edge.computeScene(scene)
        best_ObjV = edge.statisticLatency(cmni,cmpt,miga)
        engy = edge.computeEnergy(scene)
        #valOP = [best_ObjV]
        #valStack = np.hstack((valStack,valOP))
        valStack[1,index] = 1
        valStack[2,index] = 1
        valStack[3,index] = best_ObjV
        valStack[4,index] = engy
        phenStack = np.vstack((phenStack,best_Phen))

        #index+=1
        if index < lastIdx:
            nextScene = edge.updateNextScene(listScene,index)
            listScene[index+1] = nextScene

    return valStack,phenStack