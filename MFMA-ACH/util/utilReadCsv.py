import util

import pandas as pd
import numpy as np
import xlrd
import os

def readCsv(filePath):
    df = pd.read_csv(filePath)
    return df

def readExcel2Matrix(filePath):
    # pip uninstall xlrd
    # pip install xlrd==1.2.0
    # 如果安装xlrd后报错，是因为新版xlrd不兼容xls文件，要安装旧版
    table = xlrd.open_workbook(filePath).sheets()[0]
    # 可以用openpyxl代替xlrd打开.xlsx文件：
    # df=pandas.read_excel('data.xlsx',engine='openpyxl')
    row = table.nrows
    col = table.ncols
    datamatrix = np.zeros((row,col))
    for x in range(col):
        cols = np.matrix(table.col_values(x))
        datamatrix[:, x] = cols
    #print(datamatrix)
    return datamatrix

def getCellDistance(matrix,cellx,celly):
    row = cellx - 1
    col = celly - 1
    value = matrix[row,col]
    return value