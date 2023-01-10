from math import cos, sin, sqrt, radians
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import os

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

def getCircle(p1, p2, p3):
    x21 = p2.x - p1.x
    y21 = p2.y - p1.y
    x32 = p3.x - p2.x
    y32 = p3.y - p2.y
    # three colinear
    if (x21 * y32 - x32 * y21 == 0):
        return None
    xy21 = p2.x * p2.x - p1.x * p1.x + p2.y * p2.y - p1.y * p1.y
    xy32 = p3.x * p3.x - p2.x * p2.x + p3.y * p3.y - p2.y * p2.y
    y0 = (x32 * xy21 - x21 * xy32) / (2 * (y21 * x32 - y32 * x21))
    x0 = (xy21 - 2 * y0 * y21) / (2.0 * x21)
    R = ((p1.x - x0) ** 2 + (p1.y - y0) ** 2) ** 0.5
    return x0, y0, R

def draw_hex(center_x, center_y, length, ax, idx):
    start_x = center_x
    start_y = center_y
    angle = 60
    print(f'\noriginal center: {start_x:.2f}, {start_y:.2f}')

    # add center msg
    '''
    center_str = f'{idx}: ({center_x:.2f}, {center_y:.2f})'
    offsetbox = TextArea(center_str,textprops = dict(fontsize = 10))
    ab_text = AnnotationBbox(offsetbox, (center_x, center_y),
        xybox=(15, 15),
        xycoords='data',
        boxcoords=("offset points"),  # axes fraction
        frameon=False, pad=0)
    ax.add_artist(ab_text)
    '''

    plt.scatter(center_x, center_y, s=10, c="#ff6666", marker='+')

    list_hex_points = []
    bs_points = []
    for i in range(6):
        end_x = start_x + length * cos(radians(angle * i))
        end_y = start_y + length * sin(radians(angle * i))
        end_x = round(end_x, 2)
        end_y = round(end_y, 2)
        list_hex_points.append((end_x, end_y))
        plt.scatter(end_x, end_y, s=10, c="#ff6666", marker='+')
        print(f'{i+1}: {end_x:.2f}, {end_y:.2f}')

        # add center msg
        '''
        point_str = f'{i+1}'
        offsetbox = TextArea(point_str,textprops = dict(fontsize = 10))
        ab_text = AnnotationBbox(offsetbox, (end_x, end_y),
            xybox=(0, 5),
            xycoords='data',
            boxcoords=("offset points"),  # axes fraction
            frameon=False, pad=0)
        ax.add_artist(ab_text)
        '''

        # get bs points
        if (i+1)%2 == 0:
            bs_points.append((end_x, end_y))

    lines = [(list_hex_points[0], list_hex_points[1]),
            (list_hex_points[1], list_hex_points[2]),
            (list_hex_points[2], list_hex_points[3]),
            (list_hex_points[3], list_hex_points[4]),
            (list_hex_points[4], list_hex_points[5]),
            (list_hex_points[5], list_hex_points[0])]
    #for (x, y) in lines:
    #    plt.plot( (x[0], y[0]), (x[1], y[1]), color='orange', alpha=0.4 )

    return bs_points, lines

    



def draw_scatter():
    fig, ax = plt.subplots(figsize = (5,4), dpi = 150)
    # fig, ax = plt.subplots(figsize = (9,7),dpi = 150)

    ax.set_xlim(0, 2400)
    ax.set_ylim(0, 1600)

    x1 = (250, 300)
    x2 = (250, 700)
    x3 = (333, 500)
    x4 = (500, 500)

    bs_all = []
    lines_all = []
    bs_distinct = []

    # generate cell center
    cols = 7
    rows = 4
    size = 195.4
    center_list = []
    offset_ori_x = size  #size * 1.5 #445
    offset_ori_y = size * 2 #size * sqrt(3) #312.5
    idx = 0
    for c in range(cols):
            if c % 2 == 0:
                offset = size * sqrt(3) / 2
            else:
                offset = 0
            for r in range(rows):
                x = c * (size * 1.5) + offset_ori_x
                y = r * (size * sqrt(3)) - offset + offset_ori_y
                #center_list.append(Point(x, y))
                idx += 1
                bs_points, lines = draw_hex(x, y, size, ax, idx)
                for bs in bs_points:
                    bs_all.append(bs)
                for li in lines:
                    lines_all.append(li)
                
                
    
    # fig all lines
    lines_distinct = []

    for ( ((p1_x,p1_y), (p2_x,p2_y)) ) in lines_all:
        if (((p1_x,p1_y), (p2_x,p2_y)) in lines_distinct) or (((p2_x,p2_y), (p1_x,p1_y)) in lines_distinct):
            pass
        else:
            lines_distinct.append(((p1_x,p1_y), (p2_x,p2_y)))
    for ((p1_x,p1_y), (p2_x,p2_y)) in lines_distinct:
        plt.plot( (p1_x, p2_x), (p1_y, p2_y), color='orange', alpha=0.4 )
    #print(f'lines {len(lines_all)}, {len(lines_distinct)}')
    

    bs_sta = {}
    for idx, temp in enumerate(bs_all):
        #print(f'{idx} {temp}')
        if temp not in bs_sta.keys():
            bs_sta[temp] = 1
        else:
            bs_sta[temp] += 1
    # get bs with 3 overlap times
    bs_tri = []
    for k,v in bs_sta.items():
        if v == 3:
            bs_tri.append(k)

    # fig base station icon
    arr_lena = mpimg.imread('gNB.png')
    imagebox = OffsetImage(arr_lena, zoom=0.2)
    for idx, bs in enumerate(bs_tri):
        ab = AnnotationBbox(imagebox, (bs[0], bs[1]),frameon=False,pad=0)
        ax.add_artist(ab)

        offsetbox = TextArea("gNB"+str(idx+1),textprops = dict(fontsize = 5))
        ab_text = AnnotationBbox(offsetbox, (bs[0], bs[1]),
            xybox=(10, 10),
            xycoords='data',
            boxcoords=("offset points"),  # axes fraction
            frameon=False, pad=0)
        ax.add_artist(ab_text)

    # log bs locate
    log_bs = 'bs.txt'
    for idx, bs in enumerate(bs_tri):
        content = f'{bs[0]},{bs[1]}'
        with open(log_bs, 'a') as f:
            f.writelines(content+"\n")
    '''
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
    '''

    #lines = 
    # [(x1, x3), (x2, x3), (x3, x4)]
    '''
    lines = []
    for (x, y) in lines:
        #for i in [0, 1]:
        plt.plot( (x[0], y[0]), (x[1], y[1]), color='blue', alpha=0.4 )
    '''

    # circle msg
    # get center point
    '''
    x0, y0, R = getCircle(Point(250,700), Point(500,500), Point(500,875))
    print(f'compute center: {x0:.2f}, {y0:.2f}, R: {R}')
    plt.scatter(x0, y0, s=40, c="#ffff12", marker='o')

    x0, y0, R = getCircle(Point(250, 300), Point(500, 125), Point(500, 500))
    print(f'compute center: {x0:.2f}, {y0:.2f}, R: {R}')
    plt.scatter(x0, y0, s=40, c="#ffff12", marker='o')
    '''

    #x0, y0, R = getCircle(Point(750, 700), Point(750, 300), Point(500, 500))
    #print(f'compute center: {x0:.2f}, {y0:.2f}, R: {R}')
    #plt.scatter(x0, y0, s=20, c="#ffff12", marker='o')

    

    # icon
    x_list = []

    arr_lena = mpimg.imread('gNB.png')

    imagebox = OffsetImage(arr_lena, zoom=0.3)

    for idx, x in enumerate(x_list):
        ab = AnnotationBbox(imagebox, (x[0], x[1]),frameon=False,pad=0)
        ax.add_artist(ab)

        offsetbox = TextArea("gNB"+str(idx),textprops = dict(fontsize = 10))
        ab_text = AnnotationBbox(offsetbox, (x[0], x[1]),
            xybox=(15, 15),
            xycoords='data',
            boxcoords=("offset points"),  # axes fraction
            frameon=False, pad=0)
        ax.add_artist(ab_text)

    #plt.grid()
    #plt.draw()
    plt.savefig('add_picture_matplotlib_figure.png',bbox_inches='tight')
    plt.show()

draw_scatter()

#bs = 