from ctypes import sizeof
import numpy as np
import pandas as pd
import random
import sys
import time
class KMeansClusterer:
    def __init__(self,ndarray,cluster_num):
        #self.ndarray = ndarray
        arr = self.convert_pos2array(ndarray)
        self.ndarray = np.array(arr)
        self.cluster_num = cluster_num
        self.points=self.__pick_start_point(self.ndarray,cluster_num)
        self.iter_count = 0

    def convert_pos2array(self, position):
        num = len(position)
        #第一列x，第二列y
        ret = np.zeros((num, 2))
        #把(x, y, 1)转为[x, y]
        for i in range(num):
            #ret[i][0] = [position[i][0], position[i][1]]
            ret[i][0] = position[i][0]
            ret[i][1] = position[i][1]
        
        return ret

        
    def cluster(self):
        self.iter_count += 1
        result = [] #坐标
        result_idx = [] #索引
        for i in range(self.cluster_num):
            result.append([])
            result_idx.append([])
        for r_idx, item in enumerate(self.ndarray):
            distance_min = sys.maxsize
            index=-1
            for i in range(len(self.points)):                
                distance = self.__distance(item,self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
            result_idx[index] = result_idx[index] + [r_idx]
        new_center=[]
        for item in result:
            new_center.append(self.__center(item).tolist())
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points==new_center).all():
            #print("ok")
            return result, result_idx
        
        self.points=np.array(new_center)
        #print("no")
        return self.cluster()
            
    def __center(self,list):
        '''计算一组坐标的中心点
        '''
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)
    def __distance(self,p1,p2):
        '''计算两点间距
        '''
        tmp=0
        for i in range(len(p1)):
            tmp += pow(p1[i]-p2[i],2)
        return pow(tmp,0.5)
    def __pick_start_point(self,ndarray,cluster_num):
        if cluster_num <0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
     
        # 随机点的下标
        indexes=random.sample(np.arange(0,ndarray.shape[0],step=1).tolist(),cluster_num)
        points=[]
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)

"""
points = [
    [1, 2],
    [2, 1],
    [3, 1],
    [5, 4],
    [5, 5],
    [6, 5],
    [10, 8],
    [7, 9],
    [11, 5],
    [14, 9],
    [14, 14],
    ]
#print(k_means(points, 3))
kc = KMeansClusterer(np.array(points),3)
r = kc.cluster()
print(r)
"""
