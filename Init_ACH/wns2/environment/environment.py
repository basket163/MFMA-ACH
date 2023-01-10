import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
from math import sqrt

MIN_RSRP = -140

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class Environment:
    def __init__(self, h, l, sampling_time = 1, renderer = None):
        self.h = h
        self.l = l
        self.ue_list = {}
        self.bs_list = {}
        self.sampling_time = sampling_time
        self.renderer = renderer
        self.plt_run = 0
        return
    
    def add_user(self, ue):
        if ue.get_id() in self.ue_list:
            raise Exception("UE ID mismatch for ID %s", ue.get_id())
        self.ue_list[ue.get_id()] = ue
        return
    
    def remove_user(self, ue_id):
        if ue_id in self.ue_list:
            del self.ue_list[ue_id]

    def add_base_station(self, bs):
        if bs.get_id() in self.bs_list:
            raise Exception("BS ID mismatch for ID %s", bs.get_id())
        self.bs_list[bs.get_id()] = bs
        return

    def compute_rsrp(self, ue):
        rsrp = {}
        for bs in self.bs_list:
            rsrp_i = self.bs_list[bs].compute_rsrp(ue)
            if rsrp_i > MIN_RSRP or self.bs_list[bs].get_bs_type() == "sat":
                rsrp[bs] = rsrp_i
        return rsrp

    def compute_distance(self, pos1, pos2):
        distance = 0.0
        pos1_x = pos1[0]
        pos1_y = pos1[1]
        pos2_x = pos2[0]
        pos2_y = pos2[1]
        dis_x = pos1_x - pos2_x
        dis_y = pos1_y - pos2_y
        distance= sqrt((dis_x**2)+(dis_y**2))
        return distance

    def compute_nearest_bs(self, loc):
        nearest_dis = float("inf")
        nearest_bs_id = None
        for bs_id in self.bs_list:
            #print('compute bs_id: {bs_id}')
            bs = self.bs_list[bs_id]
            bs_pos = bs.position
            distance = self.compute_distance(bs_pos, loc)
            #print(f'bs {bs_id} pos:{bs_pos}, u pos: {loc}')
            if distance < nearest_dis:
                nearest_dis = distance
                nearest_bs_id = bs_id
                #print(f'{distance} < {nearest_dis}')
                #print(f'bs update {nearest_bs_id}')
        return nearest_bs_id
        
    def step(self):
        u_in_bs = []
        u_in_loc = []
        u_in_speed = []
        for ue in self.ue_list:
            self.ue_list[ue].step()
            bs = self.ue_list[ue].get_current_bs()
            loc = self.ue_list[ue].get_position() # loc = (x,y,z)
            speed = self.ue_list[ue].speed
            #print(f'a ue in bs: {bs}')
            if bs is None:
                bs = self.compute_nearest_bs(loc)
            u_in_bs.append(bs+1) # index + 1
            u_in_loc.append( loc )
            u_in_speed.append(speed)

        for bs in self.bs_list:
            self.bs_list[bs].step()
        return u_in_bs, u_in_loc, u_in_speed

    
    def render(self):
        return
        if self.renderer != None:
            return self.renderer.render(self)

    def bs_by_id(self, id):
        return self.bs_list[id]
    def ue_by_id(self, id):
        return self.ue_list[id]
    def get_sampling_time(self):
        return self.sampling_time
    def get_x_limit(self):
        return self.l
    def get_y_limit(self):
        return self.h
