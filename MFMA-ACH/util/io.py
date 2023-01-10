import os
import time
import pandas as pd
import numpy as np

import globalVar
from globalVar import enumVar
import util


def modifyFolder(old,new,timeSlot):
    folderPath = globalVar.cfg.outputPath+globalVar.pathSep#"\\"
    folderPathNew = globalVar.ResultData
    #judge folderPath
    isExists=os.path.exists(folderPathNew)
    if not isExists:
        os.makedirs(folderPathNew)
    fileTime = ioGetTimeAsFileName()
    folderResult = folderPath  + old
    curData = globalVar.get_value(enumVar.currentFolder)
    curPara = globalVar.get_value(enumVar.nowEvoPara) + "runIdx"+str(globalVar.runIdx)+ "slot"+str(timeSlot)
    prefix = util.ioGetTimeAsFileName() + '_' + curData +'_' #+ curPara + "_"
    if os.path.exists(folderResult):
        os.rename(os.path.join(folderPath,old),os.path.join(folderPathNew,prefix+new+curPara))
    #if os.path.exists(folderPath+new):
    #    print("new path success")
    #input()

def ioGetPath():
    folderPath = os.getcwd()+globalVar.pathSep#"\\"
    return folderPath

def setRunTime():
    """ record program start time. """
    runTime = time.strftime("%Y-%m%d-%H%M", time.localtime())
    return runTime

def ioGetTimeAsFileName():
    return globalVar.runTime

def ioSaveNumpy2Csv(array,name):
    np.savetxt(globalVar.ResultImg + name + '.csv',array,delimiter=',',fmt='%.4f')

def saveArray2Excel(array,folder,filename):
    df = pd.DataFrame(array)
    df.to_csv(r''+folder+filename,encoding='gbk')

def create_run_time(file_path, title):
    with open (file_path, 'w') as f:
            f.write(title)

def save_run_time(file_path, content):
    with open (file_path, 'a') as f:
        f.write(content)