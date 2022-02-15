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
    def __init__(self, eng, ID='', in_roads=[], out_roads=[]):
        """
        initialises the Analytical Agent
        :param ID: the unique ID of the agent corresponding to the ID of the intersection it represents 
        :param eng: the cityflow simulation engine
        """
        super().__init__(eng, ID)

        self.in_roads = in_roads
        self.out_roads = out_roads
        self.action_queue = queue.Queue()
        self.init_phases_vectors(eng)
        
    def act(self, eng, time):
        """
        selects the next action - phase for the agent to select along with the time it should stay on for
        :param eng: the cityflow simulation engine
        :param time: the time in the simulation, at this moment only integer values are supported
        :returns: the phase and the green time
        """

        self.update_clear_green_time(time)
        
        # self.stabilise(time)
        # if not self.action_queue.empty():
        #     phase, green_time = self.action_queue.get()
        #     return phase, int(np.ceil(green_time))

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
            green_time = max(2, int(np.max([self.movements[x].green_time for x in action.movements])))

            
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

                    if phaseID in phases_time.keys():
                        phases_time.update({phaseID : max(phases_time[phaseID], elem[1])})
                    else:
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

        for movement in [x for x in self.movements.values() if x.ID not in self.phase.movements]:              
            Q = movement.arr_rate
            if movement.last_on_time == -1:
                waiting_time = 0
            else:
                waiting_time = time - movement.last_on_time
                
            z = movement.green_time + movement.clearing_time + waiting_time
            n_crit = Q * T * ((T_max - z) / (T_max - T))

            waiting = movement.green_time * movement.max_saturation
           
            if waiting > n_crit:
                T_res = T * (1 - sum_Q / movement.max_saturation) - self.clearing_time * len(self.movements)
                green_max = (Q / movement.max_saturation) * T + (1 / len(self.movements)) * T_res
                priority_list.append((movement, green_max))
            
        priority_list = add_phase_to_queue(priority_list)
        while priority_list:
            priority_list = add_phase_to_queue(priority_list)



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
                lanes_veh_num.append(lanes_count[lane] * 5 / length)
                # lanes_veh_num.append(lanes_count[lane])
        return lanes_veh_num

    def get_in_lanes_veh_num(self, eng, lanes_veh, vehs_distance):
        """
        gets the number of vehicles on the incoming lanes of the intersection
        :param eng: the cityflow simulation engine
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
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
     
                lanes_veh_num.append(seg1 * (5 / (length/3)))
                lanes_veh_num.append(seg2 * (5 / (length/3)))
                lanes_veh_num.append(seg3 * (5 / (length/3)))

        return lanes_veh_num


    
    def init_phases_vectors(self, eng):
        """
        initialises vector representation of the phases
        :param eng: the cityflow simulation engine
        """
        idx = 1
        vec = np.zeros(len(self.phases))
        self.clearing_phase.vector = vec.tolist()
        for phase in self.phases.values():
            vec = np.zeros(len(self.phases))
            if idx != 0:
                vec[idx-1] = 1
            phase.vector = vec.tolist()
            idx+=1    
