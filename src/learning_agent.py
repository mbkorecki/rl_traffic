import cityflow
import torch
import numpy as np
import random
from operator import add

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Learning_Agent:
    
    def __init__(self, phase=0, ID='', in_roads=[], out_roads=[]):

        self.phase_to_vec = {}
        self.phase_to_movement = {}
        self.movement_to_phase = {}

        self.clearing_phase = 0
        
        self.phase = phase
        self.ID = ID
        self.in_roads = in_roads
        self.out_roads = out_roads

        self.action = 0
        self.state = None
        self.reward = 0
        self.reward_count = 0
        

        self.total_rewards = 0
        self.past_phases = []
        
        self.total_duration = {}
        
        self.max_wait_time = np.zeros(len(self.movement_to_phase))
        self.current_wait_time = np.zeros(len(self.movement_to_phase))

        
        self.action_freq = 10
        self.action_type = "act"

        self.arr_vehs_num = []
        self.dep_vehs_num = []
        self.pass_link_time = 28
        self.green_time = 10

    def init_movements_lanes_dict(self, eng):
        self.movements_lanes_dict = {}
        for roadlink in eng.get_intersection_lane_links(self.ID):
            start_lane = roadlink[1][0][0]
            end_lanes = roadlink[1][:]
            end_lanes = [x[1] for x in end_lanes]
            self.movements_lanes_dict.update({roadlink[0] : [(start_lane, end_lanes)]})


    def init_phase_to_movement(self, eng):
        movements_dict = {}
        for idx, elem in enumerate(self.movements_lanes_dict):
            movements_dict.update({elem : idx})

        empty_phases = []
        for idx, phase_tuple in enumerate(eng.get_intersection_phases(self.ID)):
            movements = []
            phases = phase_tuple[0]
            types = phase_tuple[1]

            for move in phases:
                key = tuple(move)
                move_idx = movements_dict[key]
                movements.append(move_idx)

            if types and all(x == 1 for x in types): #1 -> turn right
                self.clearing_phase = idx
            elif movements:
                self.phase_to_movement.update({idx : movements})
            else:
                empty_phases.append(idx)

                
        if not self.clearing_phase and empty_phases:
            self.clearing_phase = empty_phases[0]
            self.phase = self.clearing_phase

        for idx in range(len(movements_dict)):
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


            
        self.prev_vehs = [set() for i in range(len(self.movement_to_phase))]

        self.phases_duration = np.zeros(len(self.phase_to_movement))
        self.total_duration = {}
        for i in range(len(self.movement_to_phase)-1):
            self.total_duration.update({i : []})
        
        self.max_wait_time = np.zeros(len(self.movement_to_phase))
        self.current_wait_time = np.zeros(len(self.movement_to_phase))

        self.phases_duration = np.zeros(len(self.phase_to_vec))
        self.out_move_vehs = [[set() for i in range(len(self.movement_to_phase))]]
        self.in_move_vehs = [[set() for i in range(len(self.movement_to_phase))]]
        self.prev_vehs = [set() for i in range(len(self.movement_to_phase))]
        for i in range(len(self.movement_to_phase)):
            self.total_duration.update({i : []})
        
    def set_phase(self, eng, phase):
        eng.set_tl_phase(self.ID, self.action_to_phase[phase])
        self.phase = phase

    def observe(self, eng, time, lanes_count):
        observations = self.phase_to_vec[self.phase] + self.get_in_lanes_veh_num(eng, lanes_count) + self.get_out_lanes_veh_num(eng, lanes_count)
        # observations = self.current_wait_time.tolist() + self.phase_to_vec[self.phase] + self.get_in_lanes_veh_num(eng) + self.get_out_lanes_veh_num(eng)

        return observations

    def get_reward(self, eng, time, lanes_count):
        #compute intersection pressure
        # return -np.abs(np.sum(self.get_time_weighted_pressure(eng)))
        return -np.abs(np.sum(self.get_movements_pressure(eng, lanes_count)))

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
        

    def actAnalytical(self, eng, time):
        priority = self.get_priority_idx(eng, time)
        phases_priority = np.zeros(len(self.phase_to_vec))

        for phase in range(self.phase_to_vec):
            movements = [x for x in self.phase_to_movement[phase] if x not in self.phase_to_movement[0]]
            phase_prioirty = 0
            for move in movements:
                phase_prioirty += priority[move]

            phases_priority[phase] = phase_prioirty

        return np.argmax(phases_priority)

    def reset_veh_num(self):
        self.out_move_vehs = [[set() for i in range(12)]]
        self.in_move_vehs = [[set() for i in range(12)]]

        self.arr_vehs_num = []
        self.dep_vehs_num = []
        
        self.prev_vehs = [set() for i in range(12)]
        self.phase = self.clearing_phase

    def get_rate_pressure(self, eng, time):
        pressure = 0
        # for lane in range(12):
        #     arr_rate = self.get_arr_veh_num(eng, lane, 0, time) / (time+1)
        #     dep_rate = self.get_dep_veh_num(eng, lane, 0, time) / (time+1)
        #     pressure += (arr_rate - dep_rate)

        # self.update_clear_green_time(eng, time)
        
        # for lane in range(12):
        #     arr_rate = self.get_arr_veh_num(eng, lane, 0, time) / (time + 1)
        #     # end_time = time + 2 + self.green_times[lane] - self.pass_link_time

        #     arr = arr_rate * 20
        #     dep = (self.get_dep_veh_num(eng, lane, 0, time) / (time + 1)) * 20
        #     pressure += (arr - dep)
        
        # return pressure

        movements_pressure = []
        lanes_count = eng.get_lane_vehicle_count()
        pressure = 0
        start_time = max(1, time-60)
        
        for lane, elem in enumerate(self.movements_lanes_dict.values()):
            arr_rate = self.get_arr_veh_num(eng, lane, start_time, time) / (time - start_time)
            arr = arr_rate * 20
            pressure = lanes_count[elem[0][0]] + arr

            dep = (self.get_dep_veh_num(eng, lane, start_time, time) / (time -start_time)) * 20
            pressure -= (np.mean([lanes_count[x] for x in elem[0][1]]) + dep)
            movements_pressure.append(pressure)
        return movements_pressure

        
    def get_time_weighted_pressure(self, eng):
        movements_pressure = []
        lanes_count = eng.get_lane_vehicle_count()
        pressure = 0
        #EVERY CAR CONTRIBUTES PROPORTIONALLY TO ITS WAIT TIME
        wait_time = self.current_wait_time

        def multiplier(x):
            if x < 90: return 1
            else: return x / 90

        for idx, elem in enumerate(self.movements_lanes_dict.values()):
            pressure = lanes_count[elem[0][0]] * multiplier(wait_time[idx])
            pressure -= np.mean([lanes_count[x] for x in elem[0][1]])
            movements_pressure.append(pressure)
        return movements_pressure
    
    def get_movements_pressure(self, eng, lanes_count):
        movements_pressure = []

        pressure = 0
        for elem in self.movements_lanes_dict.values():
            pressure = int(lanes_count[elem[0][0]])
            # SHOULD IT REALLY BE MEAN!?
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
    
    def get_out_lanes_veh_num(self, eng, lanes_count):
        lanes_veh_num = []            
        for road in self.out_roads:
            lanes = eng.get_road_lanes(road)
            for lane in lanes:
                lanes_veh_num.append(lanes_count[lane])
        return lanes_veh_num

    def get_in_lanes_veh_num(self, eng, lanes_count):
        lanes_veh_num = []
        for road in self.in_roads:
            lanes = eng.get_road_lanes(road)
            for lane in lanes:
                lanes_veh_num.append(lanes_count[lane])
        return lanes_veh_num

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
        
    def get_priority_idx(self, eng, time):
        self.update_clear_green_time(eng, time)

        priority = []
        
        for idx, green_time in enumerate(self.green_times):
            if idx in self.phase_to_movement[self.phase]:
                priority.append((green_time * 2.2) / (green_time + 2))
            else:
                priority.append((green_time * 2.2) / (green_time + 2 + 2))

        return priority

    
    def get_clear_green_time(self, eng, time, phase):
        green_times = []


        lanes = self.phase_to_movement[phase]

        for lane in lanes:
            arr_rate = self.get_arr_veh_num(eng, lane, 0, time) / (time + 1)
            dep = self.get_dep_veh_num(eng, lane, 0, time)

            green_time = 0
            LHS = dep + 2.2 * green_time
            end_time = time + 2 + green_time - self.pass_link_time

            RHS = arr_rate * end_time

            while np.sqrt((LHS - RHS)**2) > 1 and LHS < RHS:
                green_time += 0.5

                LHS = dep + 2.2 * green_time
                end_time = time + 2 + green_time - self.pass_link_time

                RHS = arr_rate * end_time

            green_times.append(green_time)
            
        return green_times


    def update_clear_green_time(self, eng, time):
        self.green_times = []

        for lane in range(len(self.movement_to_phase)):
            if lane in self.phase_to_movement[0]:
                green_time = 0
            else:
                arr_rate = self.get_arr_veh_num(eng, lane, 0, time) / (time+1)
                dep = self.get_dep_veh_num(eng, lane, 0, time)

                green_time = 0
                LHS = dep + 2.2 * green_time
                end_time = time + 2 + green_time - self.pass_link_time

                RHS = arr_rate * end_time

                while np.sqrt((LHS - RHS)**2) > 1 and LHS < RHS:
                    green_time += 0.5

                    LHS = dep + 2.2 * green_time
                    end_time = time + 2 + green_time - self.pass_link_time

                    RHS = arr_rate * end_time

            self.green_times.append(green_time)


        
