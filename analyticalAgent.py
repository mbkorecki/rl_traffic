import cityflow
import numpy as np
import random

class AnalyticalAgent:
    
    def __init__(self, phase=[], ID='', in_roads=[], out_roads=[]):
        self.phase_to_vec = {-1 : [0, 0, 0, 0, 0, 0, 0, 0],
                             0 : [1, 0, 0, 0, 0, 0, 0, 0],
                             1 : [0, 1, 0, 0, 0, 0, 0, 0],
                             2 : [0, 0, 1, 0, 0, 0, 0, 0],
                             3 : [0, 0, 0, 1, 0, 0, 0, 0],
                             4 : [0, 0, 0, 0, 1, 0, 0, 0],
                             5 : [0, 0, 0, 0, 0, 1, 0, 0],
                             6 : [0, 0, 0, 0, 0, 0, 1, 0],
                             7 : [0, 0, 0, 0, 0, 0, 0, 1],
                             }

        self.phase_to_movement = {-1 :[2, 3, 6, 10],
                                  0 : [2, 3, 6, 10, 0, 7],
                                  1 : [2, 3, 6, 10, 4, 11],
                                  2 : [2, 3, 6, 10, 1, 8],
                                  3 : [2, 3, 6, 10, 5, 9],
                                  4 : [2, 3, 6, 10, 0, 1],
                                  5 : [2, 3, 6, 10, 7, 8],
                                  6 : [2, 3, 6, 10, 4, 5],
                                  7 : [2, 3, 6, 10, 9, 11]
                                  }

        self.movement_to_phase = {0 : [0, 4],
                                  1 : [2, 4],
                                  2 : [-1, 0, 1, 2, 3, 4, 5, 6, 7],
                                  3 : [-1, 0, 1, 2, 3, 4, 5, 6, 7],
                                  4 : [1, 6],
                                  5 : [3, 6],
                                  6 : [-1, 0, 1, 2, 3, 4, 5, 6, 7],
                                  7 : [0, 5],
                                  8 : [2, 5],
                                  9 : [3, 7],
                                  10 : [-1, 0, 1, 2, 3, 4, 5, 6, 7],
                                  11 : [1, 7]
                                  }


        
        self.phase = phase
        self.ID = ID
        self.in_roads = in_roads
        self.out_roads = out_roads

        self.action = 0
        self.state = 0

        self.prev_vehs = [set() for i in range(12)]

        self.arr_vehs_num = []
        self.dep_vehs_num = []

        self.past_phases = []
        
        self.phases_duration = np.zeros(9)
        self.total_duration = {}
        for i in range(12):
            self.total_duration.update({i : []})
        
        self.max_wait_time = np.zeros(12)
        self.current_wait_time = np.zeros(12)

        self.action_freq = 5
        self.action_type = "act"
        self.pass_link_time = 28

        self.total_rewards = 0

    def init_movements_lanes_dict(self, eng):
        movements_lanes_dict = {}
        for roadlink in eng.get_intersection_lane_links(self.ID):
            start_lane = roadlink[1][0][0]
            end_lanes = roadlink[1][:]
            end_lanes = [x[1] for x in end_lanes]
            movements_lanes_dict.update({roadlink[0] : [(start_lane, end_lanes)]})
        self.movements_lanes_dict = movements_lanes_dict

    def set_phase(self, eng, phase):
        #phase+1
        eng.set_tl_phase(self.ID, phase+1)
        self.phase = phase
        

    def observe(self, eng):
        observations = 0
        return observations

    def act(self, eng, state, time):
        self.get_clear_green_time(eng, time)
        print(self.green_times)
        return 0


    def get_priority_idx(self, eng, time):
        self.update_clear_green_time(eng, time)

        priority = []
        
        for green_time in self.green_times:
            priority.append((green_time * 2.2) / (green_time + 2))

        return priority
        
    def update_clear_green_time(self, eng, time):
        self.green_times = []

        for lane in range(12):
            arr_rate = self.get_arr_veh_num(eng, lane, 0, time) / time
            dep_rate = self.get_dep_veh_num(eng, lane, 0, time) / time
            
            dep = self.get_dep_veh_num(eng, lane, 0, time)

            green_time = 0
            LHS = dep + 2.2 * green_time
            end_time = time + 2 + green_time - self.pass_link_time
            
            arr_rate = self.get_arr_veh_num(eng, lane, 0, time) / time
            RHS = arr_rate * end_time

            while np.sqrt((LHS - RHS)**2) > 1 and green_time < 30 and LHS < RHS:
                green_time += 0.5
                
                LHS = dep + 2.2 * green_time
                end_time = time + 2 + green_time - self.pass_link_time

                arr_rate = self.get_arr_veh_num(eng, lane, 0, time) / time
                RHS = arr_rate * end_time

            self.green_times.append(green_time)
            
        
    def get_dep_veh_num(self, eng, movement, start_time, end_time):
        return np.sum([x[movement] for x in self.dep_vehs_num[start_time: end_time]])

    def get_arr_veh_num(self, eng, movement, start_time, end_time):
        return np.sum([x[movement] for x in self.arr_vehs_num[start_time: end_time]])
    
    def update_arr_dep_veh_num(self, eng):
        lanes_vehs = eng.get_lane_vehicles()

        vehs = []
        departed_vehs = []
        arrived_vehs = []

        for prev_vehs, elem in zip(self.prev_vehs, self.movements_lanes_dict.values()):
            current_vehs = set(lanes_vehs[elem[0][0]])
            departed_vehs.append(len(prev_vehs - current_vehs))
            arrived_vehs.append(len(current_vehs - prev_vehs))

            vehs.append(current_vehs)

        self.prev_vehs = vehs
        
        self.arr_vehs_num.append(arrived_vehs)
        self.dep_vehs_num.append(departed_vehs)


    def reset_veh_num(self):
        self.prev_vehs = [set() for i in range(12)]
