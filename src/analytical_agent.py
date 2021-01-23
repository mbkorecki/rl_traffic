import cityflow
import numpy as np
import random
import queue
import operator

from intersection import Movement, Phase
from agent import Agent

class Analytical_Agent(Agent):
    """
    The class defining an agent which controls the traffic lights using the analytical approach
    from Helbing, Lammer's works
    """
    def __init__(self, eng, ID=''):
        """
        initialises the Analytical Agent
        :param ID: the unique ID of the agent corresponding to the ID of the intersection it represents 
        :param eng: the cityflow simulation engine
        """
        super().__init__(ID)

        self.action_queue = queue.Queue()

        self.init_movements(eng)
        self.init_phases(eng)

    def set_phase(self, eng, phase):
        """
        sets the phase of the agent to the indicated phase
        :param eng: the cityflow simulation engine
        :param phase: the phase object, its ID corresponds to the phase ID in the simulation envirionment 
        """
        eng.set_tl_phase(self.ID, phase.ID)
        self.phase = phase
        

    def act(self, eng, time, waiting_vehs):
        """
        selects the next action - phase for the agent to select along with the time it should stay on for
        :param eng: the cityflow simulation engine
        :param time: the time in the simulation, at this moment only integer values are supported
        :param waiting_vehs: a dictionary with lane ids as keys and number of waiting cars as values
        :returns: the phase and the green time
        """
        self.update_priority_idx(time)
        
        if not self.action_queue.empty():
            phase, _, green_time = self.action_queue.get()
            return phase, int(np.ceil(green_time))
        else:
            self.stabilise(time, waiting_vehs)

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

    def stabilise(self, time, waiting_vehs):
        """
        Implements the stabilisation mechanism of the algorithm, updates the action queue with phases that need to be prioritiesd
        :param time: the time in the simulation, at this moment only integer values are supported
        :param waiting_vehs: a dictionary with lane ids as keys and number of waiting cars as values
        """
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
        """
        Updates the priority of the movements of the intersection, the higher priority the more the movement needs to get a green lights
        :param time: the time in the simulation, at this moment only integer values are supported
        """
        self.update_clear_green_time(time)

        for idx, movement in zip(self.movements.keys(), self.movements.values()):
            if idx in self.phase.movements:
                movement.priority = ((movement.green_time * movement.max_saturation) / (movement.green_time + 2))
            else:
                movement.priority = ((movement.green_time * movement.max_saturation) / (movement.green_time + 2 + 2))
        
    def update_clear_green_time(self, time):
        """
        Updates the green times of the movements of the intersection
        :param time: the time in the simulation, at this moment only integer values are supported
        """
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

