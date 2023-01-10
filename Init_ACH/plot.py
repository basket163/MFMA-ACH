import matplotlib.pyplot as plt
import numpy as np
import os

#from matplotlib.path import Path
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

#'通过Path类自定义marker'
#定义旋转矩阵
#matplotlib画图自定义marker_qiu_xingye的博客-CSDN博客
#https://blog.csdn.net/qiu_xingye/article/details/105918448
"""
def rot(verts, az):
    #顺时针旋转
    rad = az / 180 * np.pi
    verts = np.array(verts)
    rotMat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    transVerts = verts.dot(rotMat)
    return transVerts

iconMat = np.array([[-1.414, 1.414],
			[0, 0],
			[2.828, 2.828],
			[0, 0],
			[-1.414, 1.414]])

class CustomMarker(Path):
    def __init__(self, icon, az):
         if icon == "icon": 
             verts = iconMat  
         vertices = rot(verts, az)  
         super().__init__(vertices)
"""

def draw_scatter():
    plt.clf()

    fig, ax = plt.subplots(figsize = (1600, 1200),dpi = 150)
    ax.set_xlim(0,1600)
    ax.set_ylim(0,1200)

    plt.xlabel('x axis')
    plt.ylabel('y axis')

    #plt.xlim((0, 1600))
    #plt.ylim((0, 1200))
    xtick = range(0, 1600, 100)
    ytick= range(0, 1200, 100)
    plt.xticks(xtick)
    plt.xticks(rotation=20)
    plt.yticks(ytick)
    #plt.set_xticklabels(f'{x}' for x in xtick)
    #plt.set_yticklabels(f'{y}' for y in xtick)
    plt.grid(True,linestyle='--',color='gray',linewidth='0.5',axis='both')
    plt.tick_params(bottom='on',top='on',left='on',right='on')

    #plt.scatter(x, y, s=20, c="#ff1212", marker='o')

    '''
    plt.scatter(250, 300, s=20, c="#ff1212", marker='o')
    plt.scatter(250, 700, s=20, c="#ff1212", marker='o')
    plt.scatter(500, 125, s=20, c="#ff1212", marker='o')
    plt.scatter(500, 500, s=20, c="#ff1212", marker='o')
    plt.scatter(500, 875, s=20, c="#ff1212", marker='o')
    plt.scatter(750, 300, s=20, c="#ff1212", marker='o')
    plt.scatter(750, 700, s=20, c="#ff1212", marker='o')
    '''
    x1 = (250, 300)
    x2 = (250, 700)
    x3 = (333, 500)
    x4 = (500, 500)

    plt.scatter(x1[0], x1[1], s=20, c="#ff1212", marker='o')
    plt.scatter(x2[0], x2[1], s=20, c="#ff1212", marker='o')
    plt.scatter(250, 1100, s=20, c="#ff1212", marker='o')
    plt.scatter(500, 125, s=20, c="#ff1212", marker='o')
    plt.scatter(500, 500, s=20, c="#ff1212", marker='o')
    plt.scatter(500, 875, s=20, c="#ff1212", marker='o')
    plt.scatter(750, 300, s=20, c="#ff1212", marker='o')
    plt.scatter(750, 700, s=20, c="#ff1212", marker='o')
    plt.scatter(750, 1100, s=20, c="#ff1212", marker='o')
    plt.scatter(1000, 125, s=20, c="#ff1212", marker='o')
    plt.scatter(1000, 500, s=20, c="#ff1212", marker='o')
    plt.scatter(1000, 875, s=20, c="#ff1212", marker='o')
    plt.scatter(1250, 300, s=20, c="#ff1212", marker='o')
    plt.scatter(1250, 700, s=20, c="#ff1212", marker='o')
    plt.scatter(1250, 1100, s=20, c="#ff1212", marker='o')
    plt.scatter(1500, 125, s=20, c="#ff1212", marker='o')
    plt.scatter(1500, 500, s=20, c="#ff1212", marker='o')
    plt.scatter(1500, 875, s=20, c="#ff1212", marker='o')

    #plt.plot((250, 250), (300, 700))
    lines = [(x1, x3), (x2, x3), (x3, x4)]
    for (x, y) in lines:
        #for i in [0, 1]:
        plt.plot( (x[0], y[0]), (x[1], y[1]), color='blue', alpha=0.4 )

    # icon
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #img = plt.imread('icon.png')
    #ax.scatter(x, y, marker=img, c="red", s=1000)
    #fig, ax = plt.subplots()
    arr_lena = mpimg.imread('icon.png')
    imagebox = OffsetImage(arr_lena, zoom=1)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, (500, 500))
    ax.add_artist(ab)
    #plt.grid()
    #plt.draw()

    savepath = os.getcwd() + os.path.sep + 'bs.png' 
    plt.savefig(savepath)
    #plt.imshow(arr_lena)
    plt.show()
    


if __name__ == "__main__":
    # 运行
    draw_scatter()

    