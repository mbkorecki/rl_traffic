import random
import numpy as np
import torch
import operator
import queue

from learning_agent import Learning_Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Hybrid_Agent(Learning_Agent):

    def __init__(self, eng, ID='', in_roads=[], out_roads=[]):
        super().__init__(eng, ID, in_roads, out_roads)
        self.action_queue = queue.Queue()
        
    def act(self, net_local, state, time, lanes_count, eps = 0):
        """
        generates the action to be taken by the agent
        :param net_local: the neural network used in the decision making process
        :param state: the current state of the intersection, given by observe
        :param eps: the epsilon value used in the epsilon greedy learing
        """

        # self.stabilise(time, lanes_count)
        # if not self.action_queue.empty():
        #     phase = self.action_queue.get()
        #     return phase
    
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            net_local.eval()
            with torch.no_grad():
                action_values = net_local(state)
            net_local.train()
            return self.phases[np.argmax(action_values.cpu().data.numpy())]
        else:
            if random.random() > eps:
                #explore analytically

                # self.stabilise(time, lanes_count)
                # if not self.action_queue.empty():
                #     phase = self.action_queue.get()
                #     return phase

                self.update_clear_green_time(time)
                self.update_priority_idx(time)
                
                phases_priority = {}
                for phase in self.phases.values():
                    movements = [x for x in phase.movements if x not in self.clearing_phase.movements]
                    phase_prioirty = 0
                    for moveID in movements:
                        phase_prioirty += self.movements[moveID].priority

                    phases_priority.update({phase.ID : phase_prioirty})

                action = self.phases[max(phases_priority.items(), key=operator.itemgetter(1))[0]]
                return action
            else:
                #explore randomly
                return self.phases[random.choice(list(self.phases.keys()))]



    def stabilise(self, time, lanes_count):
        """
        Implements the stabilisation mechanism of the algorithm, updates the action queue with phases that need to be prioritiesd
        :param time: the time in the simulation, at this moment only integer values are supported
        """
        def add_phase_to_queue(priority_list):
            """
            helper function called recursievely to add phases which need stabilising to the queue
            """
            phases_score = {}
            for elem in priority_list:
                for phaseID in elem.phases:
                    if phaseID in phases_score.keys():
                        phases_score.update({phaseID : phases_score[phaseID] + 1})
                    else:
                        phases_score.update({phaseID : 1})

            if [x for x in phases_score.keys() if phases_score[x] != 0]:
                idx = max(phases_score.items(), key=operator.itemgetter(1))[0]
                self.action_queue.put(self.phases[idx])

        T = 180
        T_max = 240        
        priority_list = []
        
        for movement in [x for x in self.movements.values() if x.ID not in self.phase.movements]:              
            Q = movement.get_arr_veh_num(0, time) / time
            if movement.last_on_time == -1:
                waiting_time = 0
            else:
                waiting_time = time - movement.last_on_time

            z = 10 + movement.clearing_time + waiting_time
            n_crit = Q * T * ((T_max - z) / (T_max - T))

            waiting = sum([lanes_count[x] for x in movement.in_lanes])

           
            if waiting > n_crit:
                priority_list.append(movement)
            
        add_phase_to_queue(priority_list)
     
 


    
