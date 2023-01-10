import pandas as pd
import numpy as np
import os
import sys
from scipy import stats
from enum import IntEnum

class ResultCol(IntEnum):
	avg = 0
	var = 1
	std = 2
	pvalue = 3
	ttest = 4
	ks_statistic = 5
	bigthanPdd5isNorm = 6

def convert_dict_to_table(dict_data):
	# dict_data is {col_key: [val]}
	list_title = []
	list_val = []
	len_dict = len(dict_data)
	for key, val in dict_data.items():
		list_title.append(key)
		list_val.append(val)
	table = np.array(list_val).T.reshape((-1,len_dict))
	return table, list_title

def checkCsvExist(csvFile):
	isExists = os.path.isfile(csvFile)
	if not isExists:
		print(f'file {csvFile} not exist.')
		return False
	else:
		return True

def saveFile(df_csv, fileNewSurfix):
	filePathNoCsv = os.path.splitext(filePath)[0]
	#fileNewSurfix = '_dealed.csv'
	fileNewPath = filePathNoCsv+fileNewSurfix
	df_csv.to_csv(fileNewPath,encoding='gbk',index=False)
	print(f'saved in {fileNewPath}')
	return fileNewPath

def savePvalueFile(df_csv, oldFilename, fileNewSurfix):
	filePathNoCsv = os.path.splitext(oldFilename)[0]
	#fileNewSurfix = '_dealed.csv'
	fileNewPath = filePathNoCsv+fileNewSurfix
	df_csv.to_csv(fileNewPath,encoding='gbk')
	print(f'saved in {fileNewPath}')
	return fileNewPath

def computePvalue(ttestFile):
	for t in ttestFile:
		df = pd.read_csv(t)
		

		lenCol = len(df.columns)
		#lenRow is enum num
		lenRow = len(ResultCol)
		result = np.zeros((lenRow,lenCol))

		algLast = np.array(df[df.columns[-1]])
		avgLast = np.mean(algLast)
		varLast = np.var(algLast)
		stdLast = np.std(algLast)
		(ks_test,ks_pvalue) = stats.kstest(algLast, 'norm', (avgLast, stdLast))
		result[ResultCol['avg'],-1] = avgLast
		result[ResultCol['var'],-1] = varLast
		result[ResultCol['std'],-1] = stdLast
		result[ResultCol['ks_statistic'],-1] = ks_test
		result[ResultCol['bigthanPdd5isNorm'],-1] = ks_pvalue

		for dc in range(0,lenCol-1):
			compareAlg = np.array(df[df.columns[dc]])
			avg = np.mean(compareAlg)
			var = np.var(compareAlg)
			std = np.std(compareAlg)
			ttest, pvalue = stats.ttest_ind(compareAlg,algLast)
			(ks_test,ks_pvalue) = stats.kstest(compareAlg, 'norm', (avg, std))

			result[ResultCol['avg'],dc] = avg
			result[ResultCol['var'],dc] = var
			result[ResultCol['std'],dc] = std
			result[ResultCol['pvalue'],dc] = pvalue
			result[ResultCol['ttest'],dc] = ttest
			result[ResultCol['ks_statistic'],dc] = ks_test
			result[ResultCol['bigthanPdd5isNorm'],dc] = ks_pvalue
		df_csv = pd.DataFrame(result)

		listEnum = [i.name for i in ResultCol]
		#print(listEnum)
		#rowTitle = ['avg','var','std','P-value','t-test','ks-test','bigthanPdd5isNorm']
		rowTitle = listEnum
		colTitle = df.columns
		df_csv.columns = colTitle
		df_csv.index = rowTitle
		savePvalueFile(df_csv,t,"_pvalue.csv")


if __name__ == '__main__':
	filePath = sys.argv[1]
	if(checkCsvExist(filePath) == False):
		sys.exit()

	df = pd.read_csv(filePath)

	df.groupby('alg').mean()

	dfsort = df.sort_values(by=['file','alg','runIdx'])
	saveFile(dfsort, '_sorted.csv')

	fileUnique = dfsort['file'].unique()
	algUnique = dfsort['alg'].unique()
	lenAlg = len(algUnique)
	arrayAlg = np.array(algUnique)

	ttestFile = []

	staCol = ['percent2','f1_lenCore','Qcp_norm1','Qcp','Rho']
	
	dictStaResult = {}
	dictStaArray = {}
	dictStadf = {}
	for sc in staCol:
		dictStaResult[sc] = ''
		dictStaArray[sc] = ''
		dictStadf[sc] = ''

	for f in fileUnique:
		#rowIdx = -1
		colIdx = -1

		#20 is run times, 6 is alg num
		resultRow = 20
		resultCol = lenAlg
		
		#result_Qcp = np.zeros((20,6))
		#result_Rho = np.zeros((20,6))
		for sc in staCol:
			dictStaResult[sc] = np.zeros((resultRow,resultCol))

		for (k1,k2),subtable in dfsort.groupby(['file','alg']):
			#rowIdx += 1
			#arrayQcp = np.array(subtable['Qcp'])
			#arrayRho = np.array(subtable['Rho'])
			for sc in staCol:
				dictStaArray[sc] = np.array(subtable[sc])
			#cols = []
			if(k1 == f):
				colIdx += 1
				#cols.append(k2)
				
				#result_Qcp[:,colIdx] = arrayQcp.T
				#result_Rho[:,colIdx] = arrayRho.T
				for sc in staCol:
					dictStaResult[sc][:,colIdx] = (dictStaArray[sc]).T


		#df_Qcp = pd.DataFrame(result_Qcp)
		#df_Qcp.columns = arrayAlg
		#df_Rho = pd.DataFrame(result_Rho)
		#df_Rho.columns = arrayAlg
		for sc in staCol:
			dictStadf[sc] = pd.DataFrame(dictStaResult[sc])
			dictStadf[sc].columns = arrayAlg

		
		#dfr.index = rowTitle
		#ttestFile.append(saveFile(df_Qcp,f+"_Qcp.csv"))
		#ttestFile.append(saveFile(df_Rho,f+"_Rho.csv"))
		for sc in staCol:
			ttestFile.append(saveFile(dictStadf[sc],f+"_"+sc+".csv"))

	computePvalue(ttestFile)

	#resultGroup = dfsort.groupby(['file','alg']).groups
	#print(resultGroup)
	


	
	#dfAlg = dfNew['new'].value_counts().keys()
	#listAlg = [a for a in dfAlg]
	#listAlg.sort()
	#ttest, pvalue = stats.ttest_ind(compareAlg,lastAlg)
	
	# no use, could for validation.
	dictOperate = {'Qcp':np.mean,'Qcp':np.var,'Rho':np.mean,'Rho':np.var}
	dictRename={'Qcp':'Qcp mean','Qcp':'Qcp var','Rho':'Rho mean','Rho':'Rho var'}
	result_all =df.groupby(['file','alg'], as_index=False)[['Qcp','Rho']].agg(dictOperate).rename(columns=dictRename)
	saveFile(result_all, '_means_vars.csv')



