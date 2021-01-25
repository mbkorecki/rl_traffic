import cityflow
import torch
import numpy as np
import random
from operator import add

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
        super().__init__(ID)

        self.in_roads = in_roads
        self.out_roads = out_roads

        self.init_movements(eng)
        self.init_phases(eng)
        self.init_phases_vectors(eng)

        self.in_lanes = [x.in_lanes for x in self.movements.values()]
        self.in_lanes = set([x for sublist in self.in_lanes for x in sublist])

        self.out_lanes = [x.out_lanes for x in self.movements.values()]
        self.out_lanes = set([x for sublist in self.out_lanes for x in sublist])

        self.total_rewards = 0
        self.reward_count = 0

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

    def set_phase(self, eng, phase):
        """
        sets the phase of the agent to the indicated phase
        :param eng: the cityflow simulation engine
        :param phase: the phase object, its ID corresponds to the phase ID in the simulation envirionment 
        """
        eng.set_tl_phase(self.ID, phase.ID)
        self.phase = phase

    
    def observe(self, eng, time, lanes_count):
        """
        generates the observations made by the agents
        :param eng: the cityflow simulation engine
        :param time: the time of the simulation
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        observations = self.phase.vector + self.get_in_lanes_veh_num(eng, lanes_count) + self.get_out_lanes_veh_num(eng, lanes_count)
        return observations

    def get_reward(self, eng, time, lanes_count):
        """
        gets the reward of the agent in the form of pressure
        :param eng: the cityflow simulation engine
        :param time: the time of the simulation
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        return -np.abs(np.sum([lanes_count[x] for x in self.in_lanes]) - np.sum([lanes_count[x] for x in self.out_lanes]))

    def act(self, net_local, state, eps = 0, n_actions=8):
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
