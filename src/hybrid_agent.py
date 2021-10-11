import random
import numpy as np
import torch
import operator
import queue

from learning_agent import Learning_Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Hybrid_Agent(Learning_Agent):
    """
    The class defining the GuidedLight agent with analytic exploration and length invariant state 
    """
    def __init__(self, eng, ID='', in_roads=[], out_roads=[]):
        super().__init__(eng, ID, in_roads, out_roads)
        self.action_queue = queue.Queue()
        
    def act(self, net_local, state, time, eps = 0):
        """
        generates the action to be taken by the agent
        :param net_local: the neural network used in the decision making process
        :param state: the current state of the intersection, given by observe
        :param time: the current time of the simulation, needed for calculating analytical actions
        :param eps: the epsilon value used in the epsilon greedy learing
        """
    
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

 


    
