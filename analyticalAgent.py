import cityflow
import numpy as np
import random
import queue
import operator

class AnalyticalAgent:
    
    def __init__(self, phase=[], ID='', in_roads=[], out_roads=[]):
        self.phase_to_vec = {}
        self.phase_to_movement = {}
        self.movement_to_phase = {}
        
        self.phase = phase
        self.ID = ID
        self.in_roads = in_roads
        self.out_roads = out_roads

        self.action = 0
        self.state = 0
        self.reward = 0
        self.reward_count = 1
        
        self.arr_vehs_num = []
        self.dep_vehs_num = []

        self.past_phases = []
        self.clearing_phase = None

        self.action_freq = 10
        self.action_type = "act"
        self.movement_pass_time = {}
        self.max_saturation = 2.2
        self.max_speed = 11
        
        self.total_rewards = 0
        self.green_times = []
        self.green_time = 5

        self.action_queue = queue.Queue()
        
    def init_movements_lanes_dict(self, eng):
        self.movements_lanes_dict = {}
        self.lanes_length_dict = {}
        for roadlink in eng.get_intersection_lane_links(self.ID):
            lanes = roadlink[1][:]
            start_lanes = tuple(set([x[0] for x in lanes]))
            end_lanes = [x[1] for x in lanes]
            # if self.ID == 'cluster_100522741_596775840':
            #     print(roadlink)
            #     print(start_lanes)
            if start_lanes not in self.lanes_length_dict.keys():
                 for lane_id, length in eng.get_road_lanes_length(roadlink[0][0]):
                     self.lanes_length_dict.update({start_lanes : length})
                     # if self.ID == 'cluster_42428958_561035371':
                     #     print(lane_id, length)
                     
            self.movements_lanes_dict.update({roadlink[0] : [(start_lanes, end_lanes)]})

    def init_phase_to_movement(self, eng):
        self.movements_dict = {}
        self.movement_to_lanes = {}
        for idx, elem in enumerate(self.movements_lanes_dict):
            self.movements_dict.update({elem : idx})
            self.movement_to_lanes.update({idx : self.movements_lanes_dict[elem][0]})
            # if self.ID == 'cluster_100522741_596775840':     
            #     print(idx, elem)
            key = self.movements_lanes_dict[elem][0][0]
            self.movement_pass_time.update({idx : self.lanes_length_dict[key] / self.max_speed})

        # if self.ID == 'cluster_42428958_561035371':
        #     print(self.movement_pass_time)
        #     print(self.lanes_length_dict)
        empty_phases = []
        clear_moves = []
        
        for idx, phase_tuple in enumerate(eng.get_intersection_phases(self.ID)):
            movements = []
            phases = phase_tuple[0]
            types = phase_tuple[1]

            # if self.ID == 'cluster_100522741_596775840':
            #     print(idx, types)
                
            for move in phases:
                key = tuple(move)
                move_idx = self.movements_dict[key]
                movements.append(move_idx)
                
            if types and all(x == 1 for x in types): #1 -> turn right
                self.clearing_phase = idx
                clear_moves = movements
            if movements:
                if movements not in self.phase_to_movement.values():
                    self.phase_to_movement.update({idx : movements})
            else:
                empty_phases.append(idx)
        
        if empty_phases:
            self.clearing_phase = empty_phases[0]
            self.phase = self.clearing_phase
            self.phase_to_movement.update({self.clearing_phase : []})
        # elif self.clearing_phase is not None:
        #     self.phase = self.clearing_phase
        #     self.phase_to_movement.update({self.clearing_phase : clear_moves})


        for idx, start_lane in enumerate(self.movements_dict.keys()):
            self.movement_to_phase.update({idx : []})

    def init_phase_to_vec(self, eng):
        idx = 1
        vec = np.zeros(len(self.phase_to_movement))
        self.phase_to_vec.update({self.clearing_phase : vec.tolist()})
        for key, values in zip(self.phase_to_movement.keys(), self.phase_to_movement.values()):
            vec = np.zeros(len(self.phase_to_movement))
            if idx != 0:
                vec[idx-1] = 1
            self.phase_to_vec.update({key : vec.tolist()})
            idx+=1

            for val in values:
                self.movement_to_phase[val].append(key)

        self.action_to_phase = {}
        for idx, key in enumerate(self.phase_to_movement.keys()):
            self.action_to_phase.update({idx : key})

        self.prev_vehs = {}
        self.waiting_time = {}
        for move in self.movements_dict.values():
            self.prev_vehs.update({move : set()})
            self.waiting_time.update({move : 0})

        self.phases_duration = np.zeros(len(self.phase_to_movement))
        self.total_duration = {}
        
        self.max_wait_time = np.zeros(len(self.movement_to_phase))

        self.phases_duration = np.zeros(len(self.phase_to_vec))
        self.out_move_vehs = [[set() for i in range(len(self.movement_to_phase))]]
        self.in_move_vehs = [[set() for i in range(len(self.movement_to_phase))]]
        for i in range(len(self.movement_to_phase)):
            self.total_duration.update({i : []})
                
    def set_phase(self, eng, phase):
        eng.set_tl_phase(self.ID, phase)
        self.phase = phase
        
    def observe(self, eng):
        observations = 0
        return observations

    def act(self, eng, time, waiting_vehs):

        priority = self.get_priority_idx(eng, time)

        if not self.action_queue.empty():
            phases_priorities = self.action_queue.get()
            phase = max(phases_priorities, key=operator.itemgetter(1))[0]
            green_time = np.maximum(5, int(np.ceil(self.movement_pass_time[self.phase_to_movement[phase][0]] / 2)))

            return phase, green_time
        else:
            self.stabilise(eng, waiting_vehs, time)
        
            
        if all([x == 0 for x in self.green_times]):
            if self.phase == self.clearing_phase:
                return list(self.phase_to_movement.keys())[0], 5
            else:
                return self.phase, 5

        phases_priority = {}
        
        for phase in self.phase_to_movement.keys():
            movements = [x for x in self.phase_to_movement[phase] if x not in self.phase_to_movement[self.clearing_phase]]

            phase_prioirty = 0
            for move in movements:
                phase_prioirty += priority[move]

            phases_priority.update({phase : phase_prioirty})

        action = max(phases_priority.items(), key=operator.itemgetter(1))[0]
        return action, int(np.max([self.green_times[x] for x in self.phase_to_movement[action]]))

    def stabilise(self, eng, waiting_vehs, time):
        priority_list = []

        for phase in self.phase_to_movement.keys():
            priority = 0
            for move in self.phase_to_movement[phase]:                
                lanes = self.movement_to_lanes[move][0]

                Q = self.get_arr_veh_num(eng, move, time-300, time) / 300
                z = self.green_times[move] + 2 + self.current_wait_time[move]
                T = 120
                T_max = 121
                n_crit = Q * T * ((T_max - z) / (T_max - T))
            
                waiting = 0
                for lane in lanes:
                    waiting += waiting_vehs[lane]

                if waiting >= n_crit:
                    priority += (waiting - n_crit)
            if priority > 0:
                priority_list.append((phase, priority))

        if priority_list:
            self.action_queue.put(priority_list)

  
    def get_priority_idx(self, eng, time):
        self.update_clear_green_time(eng, time)
        priority = []

        for idx, green_time in enumerate(self.green_times):
            if idx in self.phase_to_movement[self.phase]:
                priority.append((green_time * self.max_saturation) / (green_time + 2))
            else:
                priority.append((green_time * self.max_saturation) / (green_time + 2 + 2))

        return priority
        
    def update_clear_green_time(self, eng, time):
        self.green_times = []

        for lane in range(len(self.movement_to_phase)):
            arr_rate = self.get_arr_veh_num(eng, lane, 0, time) / time
            dep = self.get_dep_veh_num(eng, lane, 0, time)

            green_time = 0
            LHS = dep + self.max_saturation * green_time
            pass_link_time = self.movement_pass_time[lane]
            
            end_time = time + 2 + green_time - pass_link_time

            RHS = arr_rate * end_time
            
            while (RHS - LHS) > 0.1 and LHS < RHS:
                green_time += 1

                LHS = dep + self.max_saturation * green_time
                end_time = time + 2 + green_time - pass_link_time

                RHS = arr_rate * end_time

            self.green_times.append(green_time)
            

    def update_wait_time(self, action, green_time):
        for move in self.movement_to_phase.keys():
            if move not in self.phase_to_movement[action]:
                self.waiting_time[move] += green_time + 2
            else:
                self.waiting_time[move] = 0
                
    def get_dep_veh_num(self, eng, movement, start_time, end_time):
        return np.sum([x[movement] for x in self.dep_vehs_num[start_time: end_time]])

    def get_arr_veh_num(self, eng, movement, start_time, end_time):
        return np.sum([x[movement] for x in self.arr_vehs_num[start_time: end_time]])
    
    def update_arr_dep_veh_num(self, eng, lanes_vehs):
        vehs = {}
        departed_vehs = {}
        arrived_vehs = {}

        for key, elem in zip(self.movements_lanes_dict.keys(), self.movements_lanes_dict.values()):
            move = self.movements_dict[key]
            # current_vehs = set(lanes_vehs[elem[0][0][0]])
            current_vehs = set()
            for lane in elem[0][0]:
                current_vehs.update(set(lanes_vehs[lane]))
            dep_vehs = len(self.prev_vehs[move] - current_vehs)
            departed_vehs.update({move : dep_vehs})
            arrived_vehs.update({move : len(current_vehs) - (len(self.prev_vehs[move]) - dep_vehs)})
            # arrived_vehs.append(len(current_vehs - prev_vehs))

            vehs.update({move : current_vehs})

        self.prev_vehs = vehs
        
        self.arr_vehs_num.append(arrived_vehs)
        self.dep_vehs_num.append(departed_vehs)

 

    def reset_veh_num(self):
        for move in self.movements_dict.values():
            self.prev_vehs.update({move : set()})
