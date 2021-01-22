import cityflow
import numpy as np
import random
import queue
import operator

from intersection import Movement, Phase


class Analytical_Agent:
    
    def __init__(self, phase=[], ID='', in_roads=[], out_roads=[]):
        
        self.ID = ID

        self.movements = {}
        self.phases = {}
        self.clearing_phase = None

        self.action_queue = queue.Queue()

        self.action = 0
        self.phase = Phase(ID="")

        self.action_freq = 10
        self.action_type = "act"
        
        self.green_times = []
        self.green_time = 5


    def init_movements(self, eng):
        for idx, roadlink in enumerate(eng.get_intersection_lane_links(self.ID)):
            lanes = roadlink[1][:]
            in_road = roadlink[0][0]
            out_road = roadlink[0][1]
            in_lanes = tuple(set([x[0] for x in lanes]))
            out_lanes = [x[1] for x in lanes]

            for _, length in eng.get_road_lanes_length(in_road):
                lane_length = length
                
            new_movement = Movement(idx, in_road, out_road, in_lanes, out_lanes, lane_length)
            self.movements.update({roadlink[0] : new_movement})
            
    def init_phases(self, eng):
        for idx, phase_tuple in enumerate(eng.get_intersection_phases(self.ID)):
            phases = phase_tuple[0]
            types = phase_tuple[1]
            empty_phases = []
            
            new_phase_moves = []
            for move, move_type in zip(phases, types):
                key = tuple(move)
                self.movements[key].move_type = move_type
                new_phase_moves.append(self.movements[key].ID)

            if types and all(x == 1 for x in types): #1 -> turn right
                self.clearing_phase = Phase(idx, new_phase_moves)

            if new_phase_moves:
                if set(new_phase_moves) not in [set(x.movements) for x in self.phases.values()]:
                    new_phase = Phase(idx, new_phase_moves)                    
                    self.phases.update({idx : new_phase})
            else:
                empty_phases.append(idx)

            if empty_phases:
                self.clearing_phase = Phase(empty_phases[0], [])
                self.phases.update({empty_phases[0] : self.clearing_phase})

        self.phase = self.clearing_phase
        temp_moves = dict(self.movements)
        self.movements.clear()
        for move in temp_moves.values():
            move.phases = []
            self.movements.update({move.ID : move})
            
        for phase in self.phases.values():
            for move in phase.movements:
                if phase.ID not in self.movements[move].phases:
                    self.movements[move].phases.append(phase.ID)

    def set_phase(self, eng, phase):
        eng.set_tl_phase(self.ID, phase.ID)
        self.phase = phase
        

    def act(self, eng, time, waiting_vehs):
        self.update_priority_idx(time)
        
        if not self.action_queue.empty():
            phase, _, green_time = self.action_queue.get()
            return phase, int(np.ceil(green_time))
        else:
            self.stabilise(waiting_vehs, time)

        if all([x.green_time == 0 for x in self.movements.values()]):
            if self.phase.ID == self.clearing_phase.ID:
                return list(self.phases.values())[0], 5
            else:
                return self.phase, 5

        phases_priority = {}
        
        for phase in self.phases.values():
            movements = [x for x in phase.movements if x not in self.clearing_phase.movements]

            phase_prioirty = 0
            for moveID in movements:
                phase_prioirty += self.movements[moveID].priority

            phases_priority.update({phase.ID : phase_prioirty})
        
        action = self.phases[max(phases_priority.items(), key=operator.itemgetter(1))[0]]
        return action, int(np.max([self.movements[x].green_time for x in action.movements]))

    def stabilise(self, waiting_vehs, time):
        priority_list = []
        sum_Q = np.sum([x.arr_rate for x in self.movements.values()])

        for phase in self.phases.values():
            priority = 0
            green_max = 0
            for moveID in phase.movements:                
                movement = self.movements[moveID]
                Q = movement.arr_rate
                z = movement.green_time + 2 + movement.waiting_time
                T = 90
                T_max = 150
                n_crit = Q * T * ((T_max - z) / (T_max - T))

                waiting = 0
                lanes = movement.in_lanes
                for lane in lanes:
                    waiting += waiting_vehs[lane]

                if waiting >= n_crit:
                    priority += (waiting - n_crit)
                    T_res = T * (1 - sum_Q / movement.max_saturation) - 2 * len(self.movements)
                    current_green_max = (Q / movement.max_saturation) * T + (1 / len(self.movements)) * T_res
                    if current_green_max > green_max:
                        green_max = current_green_max
                    
            if priority > 0:
                self.action_queue.put((phase, priority, green_max))

  
    def update_priority_idx(self, time):
        self.update_clear_green_time(time)

        for idx, movement in zip(self.movements.keys(), self.movements.values()):
            if idx in self.phase.movements:
                movement.priority = ((movement.green_time * movement.max_saturation) / (movement.green_time + 2))
            else:
                movement.priority = ((movement.green_time * movement.max_saturation) / (movement.green_time + 2 + 2))
        
    def update_clear_green_time(self, time):
        for movement in self.movements.values():
            movement.arr_rate = movement.get_arr_veh_num(0, time) / time
            dep = movement.get_dep_veh_num(0, time)

            green_time = 0
            LHS = dep + movement.max_saturation * green_time
            pass_link_time = movement.pass_time
            
            end_time = time + 2 + green_time - pass_link_time

            RHS = movement.arr_rate * end_time
            
            while (RHS - LHS) > 0.1 and LHS < RHS:
                green_time += 1

                LHS = dep + movement.max_saturation * green_time
                end_time = time + 2 + green_time - pass_link_time

                RHS = movement.arr_rate * end_time

            movement.green_time = green_time

    def update_arr_dep_veh_num(self, lanes_vehs):
        for movement in self.movements.values():
            movement.update_arr_dep_veh_num(lanes_vehs)

    def update_wait_time(self, action, green_time, waiting_vehs):
        for movement in self.movements.values():
            movement.update_wait_time(action, green_time, waiting_vehs)
            
    def reset_veh_num(self):
        for move in self.movements.values():
            move.prev_vehs = set()

    def reset_wait_times(self):
        for movement in self.movements.values():
            movement.waiting_time = 0
            movement.max_waiting_time = 0
            movement.waiting_time_list = []
