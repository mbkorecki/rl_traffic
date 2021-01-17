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
        self.pass_link_time = 0
        self.max_saturation = 2.2
        
        self.total_rewards = 0
        self.green_times = []
        self.green_time = 5

        self.action_queue = queue.Queue()
        
    def init_movements_lanes_dict(self, eng):
        self.movements_lanes_dict = {}
        for roadlink in eng.get_intersection_lane_links(self.ID):
            start_lane = roadlink[1][0][0]
            end_lanes = roadlink[1][:]
            end_lanes = [x[1] for x in end_lanes]
            self.movements_lanes_dict.update({roadlink[0] : [(start_lane, end_lanes)]})

    def init_phase_to_movement(self, eng):
        self.movements_dict = {}
        for idx, elem in enumerate(self.movements_lanes_dict):
            self.movements_dict.update({elem : idx})

        empty_phases = []
        clear_moves = []
        for idx, phase_tuple in enumerate(eng.get_intersection_phases(self.ID)):
            movements = []
            phases = phase_tuple[0]
            types = phase_tuple[1]

            for move in phases:
                key = tuple(move)
                move_idx = self.movements_dict[key]
                movements.append(move_idx)

            if types and all(x == 1 for x in types): #1 -> turn right
                self.clearing_phase = idx
                clear_moves = movements
            elif movements:
                self.phase_to_movement.update({idx : movements})
            else:
                empty_phases.append(idx)
        
        if self.clearing_phase == None and empty_phases:
            self.clearing_phase = empty_phases[0]
            self.phase = self.clearing_phase
            self.phase_to_movement.update({self.clearing_phase : []})
        elif self.clearing_phase is not None:
            self.phase = self.clearing_phase
            self.phase_to_movement.update({self.clearing_phase : clear_moves})


        for idx in range(len(self.movements_dict)):
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
        for move in self.movements_dict.values():
            self.prev_vehs.update({move : set()})
            
        self.phases_duration = np.zeros(len(self.phase_to_movement))
        self.total_duration = {}
        for i in range(len(self.movement_to_phase)-1):
            self.total_duration.update({i : []})
        
        self.max_wait_time = np.zeros(len(self.movement_to_phase))
        self.current_wait_time = np.zeros(len(self.movement_to_phase))

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
        if not self.action_queue.empty():
            move = self.action_queue.get()
            phases = self.movement_to_phase[move]
            return phases[0], 10
        else:
            self.stabilise(eng, waiting_vehs)
        
        priority = self.get_priority_idx(eng, time)
            
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

    def stabilise(self, eng, waiting_vehs):
        priority_list = []
        for key, elem in zip(self.movements_lanes_dict.keys(), self.movements_lanes_dict.values()):
            move = self.movements_dict[key]
            waiting = waiting_vehs[elem[0][0]]
            if waiting > 20:
                priority_list.append(move)

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
            end_time = time + 2 + green_time - self.pass_link_time

            RHS = arr_rate * end_time
            # if self.ID == 'cluster_42436703_596775900':
            #     print(RHS, LHS)
            while (RHS - LHS) > 0.1 and LHS < RHS:
                green_time += 1

                LHS = dep + self.max_saturation * green_time
                end_time = time + 2 + green_time - self.pass_link_time

                RHS = arr_rate * end_time

            self.green_times.append(green_time)
            
        
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
            current_vehs = set(lanes_vehs[elem[0][0]])
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
