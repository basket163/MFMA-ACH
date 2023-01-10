import numpy as np
import copy
import geatpy as ga
import yaml
import sys #sys.setrecursionlimit(1000000)
import pandas as pd
import os
import getopt
import datetime

import util
import pkgDevice
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import pkgMethod

def saveCsv(title, table, fileName):
    #np.savetxt(ioGetPath()+"ResultImg"+globalVar.pathSep+name+'.csv',array,delimiter=',')
    folderPath = globalVar.cfg.outputPath+globalVar.pathSep+globalVar.pathSep
    isExists=os.path.exists(folderPath)
    print(isExists)
    if (isExists):
    	os.makedirs(folderPath)
    path = folderPath+fileName+'.csv'
    df = pd.DataFrame(table)
    df.columns = title
    df.to_csv(r''+path,encoding='gbk')

def main(argv):
	user = 0
	slot = 0
	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:
		print ('initdata.py -u <user> -s <slot>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'test.py -i <inputfile> -o <outputfile>'
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile"):
			outputfile = arg

if __name__ == "__main__":
	main(sys.argv[1:])