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
        

    def act(self, eng, time):
        """
        selects the next action - phase for the agent to select along with the time it should stay on for
        :param eng: the cityflow simulation engine
        :param time: the time in the simulation, at this moment only integer values are supported
        :returns: the phase and the green time
        """

        self.update_clear_green_time(time)
        self.stabilise(time)

        if not self.action_queue.empty():
            phase, green_time = self.action_queue.get()
            return phase, int(np.ceil(green_time))

        if all([x.green_time == 0 for x in self.movements.values()]):
                return self.phase, 5
        
        self.update_priority_idx(time)
        phases_priority = {}
        
        for phase in self.phases.values():
            movements = [x for x in phase.movements if x not in self.clearing_phase.movements]

            phase_prioirty = 0
            for moveID in movements:
                phase_prioirty += self.movements[moveID].priority

            phases_priority.update({phase.ID : phase_prioirty})
        
        action = self.phases[max(phases_priority.items(), key=operator.itemgetter(1))[0]]
        if not action.movements:
            green_time = self.clearing_time
        else:
            green_time = int(np.max([self.movements[x].green_time for x in action.movements]))
        return action, green_time

    def stabilise(self, time):
        """
        Implements the stabilisation mechanism of the algorithm, updates the action queue with phases that need to be prioritiesd
        :param time: the time in the simulation, at this moment only integer values are supported
        """
        def add_phase_to_queue(priority_list):
            """
            helper function called recursievely to add phases which need stabilising to the queue
            """
            phases_score = {}
            phases_time = {}
            for elem in priority_list:
                for phaseID in elem[0].phases:
                    if phaseID in phases_score.keys():
                        phases_score.update({phaseID : phases_score[phaseID] + 1})
                    else:
                        phases_score.update({phaseID : 1})
                    phases_time.update({phaseID : elem[1]})

            if [x for x in phases_score.keys() if phases_score[x] != 0]:
                idx = max(phases_score.items(), key=operator.itemgetter(1))[0]
                self.action_queue.put((self.phases[idx], phases_time[idx]))
                return [x for x in priority_list if idx not in x[0].phases]
            else:
                return []

        T = 180
        T_max = 240
        sum_Q = np.sum([x.arr_rate for x in self.movements.values()])
        
        priority_list = []

        for movement in [x for x in self.movements.values() if x not in self.phase.movements]:                
            Q = movement.arr_rate
            z = movement.green_time + movement.clearing_time + movement.waiting_time

            n_crit = Q * T * ((T_max - z) / (T_max - T))

            waiting = movement.green_time * movement.max_saturation
            if waiting > n_crit:
                T_res = T * (1 - sum_Q / movement.max_saturation) -  self.clearing_time * len(self.phases)
                green_max = (Q / movement.max_saturation) * T + (1 / len(self.phases)) * T_res
                priority_list.append((movement, green_max))


        while priority_list:
            priority_list = add_phase_to_queue(priority_list)


  
    def update_priority_idx(self, time):
        """
        Updates the priority of the movements of the intersection, the higher priority the more the movement needs to get a green lights
        :param time: the time in the simulation, at this moment only integer values are supported
        """
        # additional_waiting = 0
        # for i in range(0, self.clearing_time+1):
        #     additional_waiting += np.max([self.movements[x].get_green_time(time, self.phase.movements, i) for x in self.phase.movements])
            
        # served_phase_green_time = np.max([self.movements[x].green_time for x in self.phase.movements])
        # served_phase_waiting_time = np.max([self.movements[x].get_green_time(time, []) for x in self.phase.movements])
        for idx, movement in zip(self.movements.keys(), self.movements.values()):
            if idx in self.phase.movements:
                movement.priority = ((movement.green_time * movement.max_saturation) / (movement.green_time + movement.clearing_time))
            else:
                penalty_term = movement.clearing_time
                # additional_waiting / (served_phase_green_time * movement.max_saturation)
                movement.priority = ((movement.green_time * movement.max_saturation) /
                                     (movement.green_time + movement.clearing_time + penalty_term))
        
    def update_clear_green_time(self, time):
        """
        Updates the green times of the movements of the intersection
        :param time: the time in the simulation, at this moment only integer values are supported
        """
        for movement in self.movements.values():
            green_time = movement.get_green_time(time, self.phase.movements)
            movement.green_time = green_time


