import cityflow
import numpy as np
import random

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from intersection import Movement, Phase
from agent import Agent

def DPGN(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class Policy_Agent(Agent):
    """
    An agent using policy gradients to learn
    """        
    def __init__(self, eng, ID='', state_dim=0, in_roads=[], out_roads=[]):
        """
        initialises the Policy Agent
        :param ID: the unique ID of the agent corresponding to the ID of the intersection it represents 
        :param eng: the cityflow simulation engine
        """
        super().__init__(eng, ID)

        self.in_roads = in_roads
        self.out_roads = out_roads

        self.init_phases_vectors(eng)
        self.n_actions = len(self.phases)

        hidden_sizes = [64]
        self.logits_net = DPGN(sizes=[state_dim]+hidden_sizes+[self.n_actions])
        self.optimizer = Adam(self.logits_net.parameters(), lr=1e-2)


        self.batch_obs = []          # for observations
        self.batch_acts = []         # for actions
        self.batch_weights = []      # for reward-to-go weighting in policy gradient
        self.batch_rets = []         # for measuring episode returns
        self.batch_lens = []         # for measuring episode lengths

        self.ep_rews = []            # list for rewards accrued throughout ep
        
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
    
    def observe(self, eng, time, lanes_count, lane_vehs, vehs_distance):
        """
        generates the observations made by the agents
        :param eng: the cityflow simulation engine
        :param time: the time of the simulation
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        observations = self.phase.vector + self.get_in_lanes_veh_num(eng, lane_vehs, vehs_distance) + self.get_out_lanes_veh_num(eng, lanes_count)
        return observations
    
    def reward_to_go(self, rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

    # def get_reward(self, lanes_count):
    #     """
    #     gets the reward of the agent in the form of pressure
    #     :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
    #     """
    #     return -sum(lanes_count.values())

    def get_policy(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def act(self, obs):
        return self.phases[self.get_policy(obs).sample().item()]

    def compute_loss(self, obs, act, weights):
        logp = self.get_policy(obs).log_prob(act)
        return -(logp * weights).mean()
    
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
                    if vehs_distance[veh] / length >= 0.66:
                        seg1 += 1
                    elif vehs_distance[veh] / length >= 0.33:
                        seg2 += 1
                    else:
                        seg3 +=1
     
                lanes_veh_num.append(seg1)
                lanes_veh_num.append(seg2)
                lanes_veh_num.append(seg3)

        return lanes_veh_num

