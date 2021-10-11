import cityflow
import torch
import numpy as np
import random
import operator

from learning_agent import Learning_Agent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Presslight_Agent(Learning_Agent):
    """
    The class defining an agent which controls the traffic lights using reinforcement learning approach called PressureLight
    """
    def __init__(self, eng, ID='', in_roads=[], out_roads=[]):
        """
        initialises the Learning Agent
        :param ID: the unique ID of the agent corresponding to the ID of the intersection it represents 
        :param eng: the cityflow simulation engine
        """
        super().__init__(eng, ID, in_roads, out_roads)

        
    def get_out_lanes_veh_num(self, eng, lanes_count):
        """
        gets the number of vehicles on the outgoing lanes of the intersection
        :param eng: the cityflow simulation engine
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        lanes_veh_num = []            
        for road in self.out_roads:
            lanes = eng.get_road_lanes(road)
            for lane in lanes:
                length = self.out_lanes_length[lane]
                lanes_veh_num.append(lanes_count[lane])
        return lanes_veh_num

    def get_in_lanes_veh_num(self, eng, lanes_veh, vehs_distance):
        """
        gets the number of vehicles on the incoming lanes of the intersection
        :param eng: the cityflow simulation engine
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        :param vehs_distance: dictionary with vehicle ids as keys and their distance on their current lane as value
        """
        lanes_veh_num = []
        for road in self.in_roads:
            lanes = eng.get_road_lanes(road)
            for lane in lanes: 
                length = self.in_lanes_length[lane]
                seg1 = 0
                seg2 = 0
                seg3 = 0
                vehs = lanes_veh[lane]
                for veh in vehs:
                    if veh in vehs_distance.keys():
                        if vehs_distance[veh] / length >= 0.66:
                            seg1 += 1
                        elif vehs_distance[veh] / length >= 0.33:
                            seg2 += 1
                        else:
                            seg3 +=1
     
                lanes_veh_num.append(seg1)
                lanes_veh_num.append(seg2)
                lanes_veh_num.append(seg3)

        return lanes_veh_num

