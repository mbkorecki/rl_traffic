import cityflow
import torch
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Intersection:
    
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

        self.action = None
        self.state = None
        self.reward = None

        self.out_move_vehs = [[set() for i in range(12)]]
        self.in_move_vehs = [[set() for i in range(12)]]


        self.total_rewards = 0
        self.past_phases = []
        
        self.phases_duration = np.zeros(9)
        self.total_duration = {}
        for i in range(12):
            self.total_duration.update({i : []})
        
        self.max_wait_time = np.zeros(12)
        self.current_wait_time = np.zeros(12)

        
        self.action_freq = 10
        self.action_type = "act"
        
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
        observations = self.current_wait_time.tolist() + self.phase_to_vec[self.phase] + self.get_in_lanes_veh_num(eng) + self.get_out_lanes_veh_num(eng)
        return observations

    def get_reward(self, eng):
        #compute intersection pressure
        # return -np.abs(self.get_total_pressure(eng))
        return -np.abs(np.sum(self.get_time_weighted_pressure(eng)))
        # return -np.abs(np.sum(self.get_movements_pressure(eng)))

    def act(self, net_local, state, eps = 0, n_actions=8):
        #Epsilon -greedy action selction
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            net_local.eval()
            with torch.no_grad():
                action_values = net_local(state)
            net_local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(n_actions))


    def reset_veh_num(self):
        self.out_move_vehs = [[set() for i in range(12)]]
        self.in_move_vehs = [[set() for i in range(12)]]

    def get_arr_dep_veh_num(self, eng):
        lanes_vehs = eng.get_lane_vehicles()
        prev_dep_vehs = self.out_move_vehs[-1]
        prev_arr_vehs = self.in_move_vehs[-1]
        vehs = []
        departed_vehs = []
        arrived_vehs = []
        
        for idx, elem in enumerate(self.movements_lanes_dict.values()):
            current_vehs = set(lanes_vehs[elem[0][0]])
            departed_vehs.append(len(prev_dep_vehs[idx] - current_vehs))
            arrived_vehs.append(len(current_vehs - prev_arr_vehs[idx]))
            
            vehs.append(current_vehs)

        self.out_move_vehs.append(vehs)
        self.in_move_vehs.append(vehs)
        return arrived_vehs, departed_vehs

    def get_time_weighted_pressure(self, eng):
        movements_pressure = []
        lanes_count = eng.get_lane_vehicle_count()
        pressure = 0
        wait_time = self.max_wait_time

        def multiplier(x):
            if x < 90: return 1
            else: return x / 90


        for idx, elem in enumerate(self.movements_lanes_dict.values()):
            pressure = lanes_count[elem[0][0]] * multiplier(wait_time[idx])
            pressure -= np.mean([lanes_count[x] for x in elem[0][1]])
            movements_pressure.append(pressure)
        return movements_pressure
    
    def get_movements_pressure(self, eng):
        movements_pressure = []
        # lanes_scount = eng.get_lane_waiting_vehicle_count()
        lanes_count = eng.get_lane_vehicle_count()

        pressure = 0
        for elem in self.movements_lanes_dict.values():
            pressure = int(lanes_count[elem[0][0]])
            pressure -= int(np.mean([lanes_count[x] for x in elem[0][1]]))
            movements_pressure.append(pressure)
        return movements_pressure

    def get_movements_demand(self, eng):
        movements_demand = []
        lanes_count = eng.get_lane_waiting_vehicle_count()        

        for elem in self.movements_lanes_dict.values():
            demand = int(lanes_count[elem[0][0]])
            movements_demand.append(demand)
        return movements_demand
    
    def get_out_lanes_veh_num(self, eng):
        lanes_vehs_dict = eng.get_lane_vehicle_count()
        lanes_veh_num = []
        for road in self.out_roads:
            lanes = eng.get_road_lanes(road)
            for lane in lanes:
                lanes_veh_num.append(lanes_vehs_dict[lane])
        return lanes_veh_num

    def get_in_lanes_veh_num(self, eng):
        lanes_vehs_dict = eng.get_lane_vehicle_count()
        lanes_veh_num = []
        for road in self.in_roads:
            lanes = eng.get_road_lanes(road)
            for lane in lanes:
                lanes_veh_num.append(lanes_vehs_dict[lane])
        return lanes_veh_num
