import cityflow
import torch
import numpy as np
import random
import operator

from intersection import Movement, Phase
from agent import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Learning_Agent(Agent):
    """
    The class defining an agent which controls the traffic lights using reinforcement learning approach called PressureLight
    """
    def __init__(self, eng, ID='', in_roads=[], out_roads=[]):
        """
        initialises the Learning Agent
        :param ID: the unique ID of the agent corresponding to the ID of the intersection it represents 
        :param eng: the cityflow simulation engine
        """
        super().__init__(eng, ID)

        self.in_roads = in_roads
        self.out_roads = out_roads

        self.init_phases_vectors(eng)

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
    
    def observe(self, eng, time, lanes_count):
        """
        generates the observations made by the agents
        :param eng: the cityflow simulation engine
        :param time: the time of the simulation
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        observations = self.phase.vector + self.get_in_lanes_veh_num(eng, lanes_count) + self.get_out_lanes_veh_num(eng, lanes_count)
        return observations

    def act(self, net_local, state, time, eps = 0, n_actions=8):
        """
        generates the action to be taken by the agent
        :param net_local: the neural network used in the decision making process
        :param state: the current state of the intersection, given by observe
        :param eps: the epsilon value used in the epsilon greedy learing
        :param n_actions: number of actions to choose from
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
                return self.phases[random.choice(np.arange(n_actions))]
    
    
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
                lanes_veh_num.append(lanes_count[lane])
        return lanes_veh_num

    def get_in_lanes_veh_num(self, eng, lanes_count):
        """
        gets the number of vehicles on the incoming lanes of the intersection
        :param eng: the cityflow simulation engine
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        lanes_veh_num = []
        for road in self.in_roads:
            lanes = eng.get_road_lanes(road)
            for lane in lanes:
                lanes_veh_num.append(lanes_count[lane])
        return lanes_veh_num

