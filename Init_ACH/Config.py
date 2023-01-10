import os
import yaml
import numpy as np
import pickle

class Config(object):

    def __init__(self, cfg_file=''):
        #print("config init.")

        if cfg_file == '':
            cfg_file = 'cfg.yml'
            print(f'config is null, set to default value: {cfg_file}.')
            #cfg_file = r'./globalCfg.yml'
        isExists=os.path.exists(cfg_file)
        if not isExists:
            input(f'config file {cfg_file} not exists.')
        else:
            print(f'load config file {cfg_file}.')
        f = open(cfg_file)
        configFile = yaml.safe_load(f)

        self.pathSep = ''

        self.outputCustom =  configFile["outputCustom"]
        self.outputDisk = configFile["outputDisk"]
        self.outputFolder = configFile["outputFolder"]
        # default output path
        self.outputPath = os.getcwd()
        #print(self.outputPath)
        if self.outputCustom == 1:
            self.outputPath = self.outputDisk + os.path.sep + self.outputFolder
            if not os.path.exists(self.outputPath):
                os.makedirs(self.outputPath)
        #print(self.outputPath)
        #input()

        

        #self.color = configFile["color"]
        #self.marker = configFile["marker"]
        

        self.init_var()

    def init_var(self):
        pass

def save_pkl(cacheName,cacheObject, folderPath):
    cachePath = folderPath + os.path.sep + cacheName + ".pkl"
    f=open(cachePath,'wb')  
    pickle.dump(cacheObject,f,-1)
    f.close()
    print(cachePath)

def load_pkl(pkl_path):
	#folderPath = cacheGetPath()
	#cachePath = folderPath + "pCache" + os.path.sep + cacheName + ".pkl"
	if os.path.exists(pkl_path):
		f = open(pkl_path,'rb+')
		cc = pickle.load(f)
		f.close()
		return cc
	else:
		print(pkl_path + "pkl file not exist.")
		return ""
