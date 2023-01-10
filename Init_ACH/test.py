from wns2.basestation.nrbasestation import NRBaseStation
from wns2.basestation.satellitebasestation import SatelliteBaseStation
from wns2.userequipment.userequipment import UserEquipment
from wns2.environment.environment import Environment
from wns2.renderer.renderer import CustomRenderer
import numpy.random as random
import logging
import pandas as pd
import os
import numpy as np

from math import cos, sin, sqrt, radians
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from kmeans import KMeansClusterer

from Config import Config, load_pkl, save_pkl

def draw_hex(center_x, center_y, length, ax, idx):
    start_x = center_x
    start_y = center_y
    angle = 60
    #print(f'\noriginal center: {start_x:.2f}, {start_y:.2f}')

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

    #plt.scatter(center_x, center_y, s=10, c="#ff6666", marker='+')

    list_hex_points = []
    bs_points = []
    for i in range(6):
        end_x = start_x + length * cos(radians(angle * i))
        end_y = start_y + length * sin(radians(angle * i))
        end_x = round(end_x, 2)
        end_y = round(end_y, 2)
        list_hex_points.append((end_x, end_y))
        #plt.scatter(end_x, end_y, s=10, c="#ff6666", marker='+')
        #print(f'{i+1}: {end_x:.2f}, {end_y:.2f}')

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

def fig_slot(x_lim, y_lim, slot_idx, u_in_bs_list, u_in_loc_list, u_in_speed, k_group, k_group_idx, cfg):
    #plt.figure(2)
    #fig, ax = plt.subplots(figsize = (21, 16), dpi = 300)
    ##ax = plt.figure(2) #(figsize = (5,4), dpi = 300)
    # dpi default 72, if change dpi, figure will shrink
    #246,216,192 灰黄
    #254,212,152 蛋黄
    #246,176,164 浅绯红
    #223,141,143 绯红
    #126,208,248 浅蓝
    #126,162,237 蓝
    #145,170,157 浅绿
    #92, 147,148 绿
    #158,148,182 浅紫
    #177,119,222 紫
    #242,5,5 红
    #166,3,3 大红
    #115,85,74棕
    #52,31,22 深棕

    #F6D8C0	灰黄
    #FED498	蛋黄
    #F6B0A4	浅绯红
    #DF8D8F	绯红
    #7ED0F8	浅蓝
    #7EA2ED	蓝
    #91AA9D	浅绿
    #5C9394	绿
    #9E94B6	浅紫
    #B177DE	紫
    #F20505	红
    #A60303	大红
    #73554A	棕
    #341F16	深棕
    
    dict_color = {
        'my_light_green':'#91AA9D',
        'my_green':'#5C9394'
    }


    #k-means分组的数量
    len_group = len(k_group)
    if len_group>5:
        print("k-means groups > 5")
        input("ctrl+c to exit.")
    # 基站是蓝色，用户是绿色，所以不能用blue和green
    color_list_all = ['#32c18f', "blue", "orange", "purple", "brown"]
    color_list_kmeans = color_list_all[:len_group]

    fig = plt.figure(figsize=(22,16), dpi=72)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    #y_min, y_max = ax.get_ylim()
    #ticks = [(tick - y_min)/(y_max - y_min) for tick in ax.get_yticks()]

    fig_title = f'Slot {slot_idx}'
    plt.title("",fontsize = 50)
    #plt.title(fig_title,fontsize = 50)
    pad = 20
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    x_tick = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400]#np.arange(0, x_lim, x_lim/10)
    y_tick = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600]#np.arange(0, y_lim, y_lim/10)
    #ax.set_xticks(x_tick)
    #ax.set_yticks(y_tick)
    #ax.set_xticklabels(fontsize = 5)
    #ax.set_yticklabels(fontsize = 5)
    ax.set_xticklabels(x_tick,fontsize = 25)
    ax.set_yticklabels(y_tick,fontsize = 25) # family = 'SimHei',
    #plt.xticks(my_x1)
    #ax.set_xlabel("m",fontsize=15)
    plt.xlabel('(m)', horizontalalignment='right', x=1.0, labelpad=0, fontsize = 25)
    plt.ylabel('(m)', horizontalalignment='left', y=1.02, labelpad=-40, fontsize = 25, rotation=0)

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
        plt.plot( (p1_x, p2_x), (p1_y, p2_y), color='orange', alpha=1.0, linestyle='--', lw='1.0' )
    #print(f'lines {len(lines_all)}, {len(lines_distinct)}')
    
    bs_x_min = None
    bs_x_max = None
    bs_y_min = None
    bs_y_max = None
    bs_sta = {}
    for idx, temp in enumerate(bs_all):
        #print(f'{idx} {temp}')
        if temp not in bs_sta.keys():
            bs_sta[temp] = 1
        else:
            bs_sta[temp] += 1
        '''
        if idx == 0:
            bs_x_min = temp[0]
            bs_x_max = temp[0]
            bs_y_min = temp[1]
            bs_y_max = temp[1]
        else:
            bs_x_min = temp[0] if temp[0]<bs_x_min else bs_x_min
            bs_x_max = temp[0] if temp[0]>bs_x_max else bs_x_max
            bs_y_min = temp[1] if temp[1]<bs_y_min else bs_y_min
            bs_y_max = temp[1] if temp[1]>bs_y_max else bs_y_max
        '''
    # print(f'bs_x_min {bs_x_min} bs_x_max {bs_x_max} bs_y_min {bs_y_min} bs_y_max {bs_y_max}')
    # bs_x_min 0.0 bs_x_max 2051.7 bs_y_min 52.36 bs_y_max 1575.35
    # get bs with 3 overlap times
    bs_tri = []
    for k,v in bs_sta.items():
        if v == 3:
            bs_tri.append(k)

    # log bs locate
    '''
    log_bs = 'bs.txt'
    for idx, bs in enumerate(bs_tri):
        content = f'{bs[0]},{bs[1]}'
        with open(log_bs, 'a') as f:
            f.writelines(content+"\n")
    '''


    # fig ue
    image_ue = mpimg.imread('ue.png')
    image_car = mpimg.imread('CarGrey.png')
    imagebox_ue = OffsetImage(image_ue, zoom=0.5)
    imagebox_car = OffsetImage(image_car, zoom=0.3)
    man_offset = (0, -30)
    man_offset_right = (46, -5)
    car_offset = (0, 25)
    car_offset_right = (72, 0)
    ini_offset = (0, 0)
    pad = 55
    for u_idx, u in enumerate(u_in_loc_list):
        #判断用户在哪一个kmeans分组，并获取颜色
        u_label_color = "gray"
        u_group_id = -1
        for g_idx in range(len_group):
            if u_idx in k_group_idx[g_idx]:
                u_label_color = color_list_kmeans[g_idx]
                u_group_id = g_idx + 1

        #plt.scatter(u[0], u[1], s=5, c="#559922", marker='s')
        speed = u_in_speed[u_idx]
        loc_x = u[0]
        loc_y = u[1]
        if u[0] < 30:
            loc_x += 30
        if u[1] < 30:
            loc_y += 30
        if speed > 10:
            ab_ue = AnnotationBbox(imagebox_car, (loc_x, loc_y),frameon=False,pad=0)
            ini_offset = car_offset
            # move label away from frame
            if u[0] < pad or u[1] < pad or u[1] > 1280:
                ini_offset = car_offset_right
            '''
            if :
                ini_offset = car_offset_right
            if :
                ini_offset = car_offset_right
            '''
        else:
            ab_ue = AnnotationBbox(imagebox_ue, (loc_x, loc_y),frameon=False,pad=0)
            ini_offset = man_offset
            # move label away from frame
            if u[0] < pad or u[1] < pad or u[1] > 1280:
                ini_offset = man_offset_right
        ax.add_artist(ab_ue)
        # msg
        u_group_tip = f'[G{str(u_group_id)}]'
        offsetbox_ue = TextArea("UE"+str(u_idx+1)+u_group_tip,textprops = dict(fontsize = 25, color=u_label_color))
        ab_text_ue = AnnotationBbox(offsetbox_ue, (loc_x, loc_y),
            xybox=ini_offset,
            xycoords='data',
            boxcoords=("offset points"),  # axes fraction
            frameon=False, pad=0)
        ax.add_artist(ab_text_ue)

    # fig base station icon
    arr_lena = mpimg.imread('gNB.png')
    imagebox = OffsetImage(arr_lena, zoom=1, alpha=0.4)
    for idx, bs in enumerate(bs_tri):
        ab = AnnotationBbox(imagebox, (bs[0], bs[1]),frameon=False,pad=0)
        ax.add_artist(ab)

        offsetbox = TextArea("gNB"+str(idx+1),textprops = dict(fontsize = 25, color="gray"))
        ab_text = AnnotationBbox(offsetbox, (bs[0], bs[1]),
            xybox=(0, 40),
            xycoords='data',
            boxcoords=("offset points"),  # axes fraction
            frameon=False, pad=0)
        ax.add_artist(ab_text)

    #plt.grid()
    #plt.draw()
    if fig_title == "":
        fig_title = f'slot_{slot_idx}'
    plt.savefig(cfg.outputPath+os.path.sep+fig_title+'.png') # ,bbox_inches='tight'
    #plt.show()

def save_group(k_group, k_group_idx, slot_idx, path):
    save_pkl("group_"+str(slot_idx),k_group_idx,path)

#program start 
cfg = Config()

logger = logging.getLogger()
#logger.setLevel(level=logging.INFO)
logger.setLevel(level=logging.WARNING)

#设置用户人数
user_num = 50
#设置分组数量
group_num=5
#设置人车比例
man_ratio = 0.5
counter = 600 #1000
# bs_x_min 0.0 bs_x_max 2051.7 bs_y_min 52.36 bs_y_max 1575.35
x_lim = 2200 # 1000
y_lim = 1600 # 1000
env = Environment(x_lim, y_lim, renderer = CustomRenderer())
user_type = np.zeros((user_num, 6)) 
# 6: userId   userType(1或2)	userInCell(根据位置得到)	profileInServ(与InCell相同)	queue(初始为0)	requestUnit(5或10)

comp_ratio=0.5
#随机生成用户位置，按比例设置用户类型
for i in range(0, user_num):
    pos = (random.rand()*x_lim, random.rand()*y_lim, 1)
    ue_speed = 1.2 #(m/s)
    user_type[i, 0] = i
    if i >= user_num*man_ratio:
        ue_speed = 12 #(m/s)
        user_type[i, 1] = 2 #type=2表示车
    else:
        ue_speed = 1.2 #(m/s)
        user_type[i, 1] = 1 #type=1表示人
    env.add_user(UserEquipment(env, i, 25, pos, speed=ue_speed, direction = random.randint(0, 360), _lambda_c=5, _lambda_d = 15))

    #用户排队0
    user_type[i, 4] = 0
    #用户计算量
    user_type[i, 5] = np.random.randint(5, 11)
    #if np.random.random() < comp_ratio:
        



bs_parm_file = 'bs.csv'
bs_parm = []
df = pd.read_csv(bs_parm_file)
for index, row in df.iterrows():
    pos = (row.pos_x, row.pos_y, row.pos_z)
    bs_dict = {
        "pos": pos,
        "freq": row.freq,
        "numerology": row.numerology, 
        "power": row.power,
        "gain": row.gain,
        "loss": row.loss,
        "bandwidth": row.bandwidth,
        "max_bitrate": row.max_bitrate
    }
    bs_parm.append(bs_dict)
#print(bs_parm)
#input('bs_parm')


bs_parm1 =[{"pos": (500, 500, 30),
    "freq": 800,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 20,
    "max_bitrate": 1000},
    
    #BS2
    {"pos": (250, 300, 30),
    "freq": 1700,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    "max_bitrate": 1000},

    #BS3
    {"pos": (500, 125, 30),
    "freq": 1900,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    #15
    "max_bitrate": 1000},

    #BS4
    {"pos": (750, 300, 30),
    "freq": 2000,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 25,
    "max_bitrate": 1000},
    
    #BS5
    {"pos": (750, 700, 30),
    "freq": 1700,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    "max_bitrate": 1000},

    #BS6
    {"pos": (500, 875, 30),
    "freq": 1900,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    "max_bitrate": 1000},

    #BS7
    {"pos": (250, 700, 30),
    "freq": 2000,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 25,
    "max_bitrate": 1000}]


for i in range(len(bs_parm)):
    env.add_base_station(NRBaseStation(env, i, bs_parm[i]["pos"], bs_parm[i]["freq"], bs_parm[i]["bandwidth"], bs_parm[i]["numerology"], bs_parm[i]["max_bitrate"], bs_parm[i]["power"], bs_parm[i]["gain"], bs_parm[i]["loss"]))

#env.add_base_station(SatelliteBaseStation(env, i+1, (250, 500, 35786000)))

slot_u_in_bs = []
slot_u_in_loc = []
slot_u_in_speed = []
title = range(1, counter+1)
slot_idx = 0
iter_count_list = []
while counter != 0:
    env.render()
    u_in_bs, u_in_loc, u_in_speed = env.step()
    #u_in_loc是这些点的坐标
    #求出kmeans
    kc = KMeansClusterer(u_in_loc,group_num)
    k_group, k_group_idx = kc.cluster()
    #print(f'iter_count: {kc.iter_count}')
    iter_count_list.append(kc.iter_count)
    #input()
    
    if counter % 60 == 0:
        slot_idx += 1
        #print(f'\n\n###\n{u_in_bs}\n###\n\n')
        print(f'\n###\nslot: {slot_idx}, counter left: {counter}\n###\n')
        slot_u_in_bs.append(u_in_bs)
        slot_u_in_loc.append(u_in_loc)
        slot_u_in_speed.append(u_in_speed)

        # fig in each slot
        fig_slot(x_lim, y_lim, slot_idx, u_in_bs, u_in_loc, u_in_speed, k_group, k_group_idx, cfg)
        save_group(k_group, k_group_idx, slot_idx, cfg.outputPath)

        # 初始位置保存到文件user_type, 后续移动位置保存到move
        if slot_idx == 1:
            for u_id in range(user_num):
                user_type[u_id, 2] = slot_u_in_bs[0][u_id]
                user_type[u_id, 3] = slot_u_in_bs[0][u_id]
    counter -= 1

#print('bs of all users')
#print(slot_u_in_bs)
#print('loc of all users')
#print(slot_u_in_loc)
#input('saving')
print(iter_count_list)
print(f'iter_count_list mean: {np.mean(iter_count_list)}, min: {np.min(iter_count_list)}, max: {np.max(iter_count_list)}')

mat = np.mat(slot_u_in_bs)
mat_t = mat.T
#print(mat_t)
#print()
df = pd.DataFrame(mat_t)
#索引列代表用户id
df.index += 1
df.index.name = "userId"
#列名
df.columns=[f'slot{str(i)}' for i in range(1,slot_idx+1)]
#print(df)
#path = os.getcwd() + os.path.sep + 'temp_move.csv'
path = cfg.outputPath + os.path.sep + 'move.csv'
df.to_csv(r''+path, encoding='gbk', index=True)

df_type = pd.DataFrame(user_type)
#列名
df_type.columns=['userId','userType','userInCell','profileInServ','queue','requestUnit']
#第一列用户id从0开始，需要整列加1
df_type['userId']  = df_type['userId'].apply(lambda x : x+1)

path_type = cfg.outputPath + os.path.sep + 'user.csv'
df_type.to_csv(r''+path_type, encoding='gbk', index=False)

# in this slot, fig all bs and users.


