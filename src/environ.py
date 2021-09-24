import cityflow
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from dqn import DQN, ReplayMemory, optimize_model
from learning_agent import Learning_Agent
from analytical_agent import Analytical_Agent
from demand_agent import Demand_Agent
from hybrid_agent import Hybrid_Agent
from presslight_agent import Presslight_Agent
from fixed_agent import Fixed_Agent
from random_agent import Random_Agent

# from policy_agent import DPGN, Policy_Agent

class Environment:
    """
    The class Environment represents the environment in which the agents operate in this case it is a city
    consisting of roads, lanes and intersections which are controled by the agents
    """
    def __init__(self, args, n_actions=9, n_states=44):
        """
        initialises the environment with the arguments parsed from the user input
        :param args: the arguments input by the user
        :param n_actions: the number of possible actions for the learning agent, corresponds to the number of available phases
        :param n_states: the size of the state space for the learning agent
        """
        self.eng = cityflow.Engine(args.sim_config, thread_num=8)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.update_freq = args.update_freq      # how often to update the network
        self.batch_size = args.batch_size
        
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay= args.eps_decay
        self.eps_update = args.eps_update
        
        self.eps = self.eps_start

        self.agents = []

        self.agents_type = args.agents_type
        if self.agents_type == 'analytical':
            self.step = self.analytical_step
        elif self.agents_type == 'learning' or  self.agents_type == 'hybrid' or self.agents_type == 'presslight':
            self.step = self.learning_step
        elif self.agents_type == 'demand' or self.agents_type == 'fixed' or self.agents_type == 'random':
            self.step = self.demand_step
        # elif self.agents_type == 'policy':
        #     self.step = self.policy_step
        else:
            raise Exception("The specified agent type:", args.agents_type, "is incorrect, choose from: analytical/policy/learning/demand/hybrid")  
        
        agent_ids = [x for x in self.eng.get_intersection_ids() if not self.eng.is_intersection_virtual(x)]
        for agent_id in agent_ids:
            if self.agents_type == 'analytical':
                new_agent = Analytical_Agent(self.eng, ID=agent_id, in_roads=self.eng.get_intersection_in_roads(agent_id), out_roads=self.eng.get_intersection_out_roads(agent_id))
            elif self.agents_type == 'learning':
                new_agent = Learning_Agent(self.eng, ID=agent_id, in_roads=self.eng.get_intersection_in_roads(agent_id), out_roads=self.eng.get_intersection_out_roads(agent_id))
            elif self.agents_type == 'demand':
                new_agent = Demand_Agent(self.eng, ID=agent_id)
            elif self.agents_type == 'hybrid':
                new_agent = Hybrid_Agent(self.eng, ID=agent_id, in_roads=self.eng.get_intersection_in_roads(agent_id), out_roads=self.eng.get_intersection_out_roads(agent_id))
            elif self.agents_type == 'presslight':
                new_agent = Presslight_Agent(self.eng, ID=agent_id, in_roads=self.eng.get_intersection_in_roads(agent_id), out_roads=self.eng.get_intersection_out_roads(agent_id))
            elif self.agents_type == 'fixed':
                new_agent = Fixed_Agent(self.eng, ID=agent_id)
            # elif self.agents_type == 'policy':
            #     new_agent = Policy_Agent(self.eng, ID=agent_id, state_dim=n_states, in_roads=self.eng.get_intersection_in_roads(agent_id), out_roads=self.eng.get_intersection_out_roads(agent_id))
            elif self.agents_type == 'random':
                new_agent = Random_Agent(self.eng, ID=agent_id)
            else:
                raise Exception("The specified agent type:", args.agents_type, "is incorrect, choose from: analytical/learning/demand/hybrid")  


            if len(new_agent.phases) <= 1:
                keys = [x for x in new_agent.phases.keys()]
                if keys == []:
                    new_agent.set_phase(self.eng, new_agent.clearing_phase)
                else:
                    new_agent.set_phase(self.eng, new_agent.phases[0])
            else:
                self.agents.append(new_agent)

        self.action_freq = 10   #typical update freq for agents

        
        self.n_actions = len(self.agents[0].phases)
        self.n_states = n_states

        if args.load:
            self.local_net = DQN(n_states, self.n_actions, seed=2).to(self.device)
            self.local_net.load_state_dict(torch.load(args.load, map_location=torch.device('cpu')))
            self.local_net.eval()
            
            self.target_net = DQN(n_states, self.n_actions, seed=2).to(self.device)
            self.target_net.load_state_dict(torch.load(args.load, map_location=torch.device('cpu')))
            self.target_net.eval()
        else:
            self.local_net = DQN(n_states, self.n_actions, seed=2).to(self.device)
            self.target_net = DQN(n_states, self.n_actions, seed=2).to(self.device)

        self.optimizer = optim.Adam(self.local_net.parameters(), lr=args.lr, amsgrad=True)
        self.memory = ReplayMemory(self.n_actions, batch_size=args.batch_size)

        self.policy_memory = []

        # self.seed = torch.manual_seed(2)
        # hidden_sizes = [128, 64]
        # self.logits_net = DPGN(sizes=[self.n_states]+hidden_sizes+[self.n_actions])
        # self.pol_opt = Adam(self.logits_net.parameters(), lr=5e-4, amsgrad=True)

        # self.value_net = DPGN(sizes=[self.n_states]+hidden_sizes+[1])
        # self.val_opt = Adam(self.value_net.parameters(), lr=1e-3, amsgrad=True)


        self.agents = [x for x in self.agents if all(x.in_lanes_length.values()) > 0]
        self.agents = [x for x in self.agents if all(x.out_lanes_length.values()) > 0]

        for agent in self.agents:
            agent.init_neighbours(self.agents)
        
        print(len(self.agents))



        
    def analytical_step(self, time, done):
        """
        represents a single step of the simulation for the analytical agent
        :param time: the current timestep
        :param done: flag indicating weather this has been the last step of the episode, used for learning, here for interchangability of the two steps
        """
        print(time)
        lane_vehs = self.eng.get_lane_vehicles()
        lanes_count = self.eng.get_lane_vehicle_count()
        veh_distance = self.eng.get_vehicle_distance()
        waiting_vehs = None

        for agent in self.agents:
            
            agent.update_arr_dep_veh_num(self.eng, lane_vehs)
            
            if time % (10 + agent.clearing_time) == 0:
                agent.total_rewards += agent.get_reward(lanes_count)
                agent.reward_count += 1
            if time % agent.action_freq == 0:
                if agent.action_type == "reward":
                    next_state = torch.FloatTensor(agent.observe(self.eng, time, lanes_count, lane_vehs, veh_distance)).unsqueeze(0)
                    self.memory.add(agent.state, agent.action.ID, 0, next_state, False)                    
                    agent.action_type = "act"
                                    
                if agent.action_type == "act":
                    agent.action, agent.green_time = agent.act(self.eng, time)
               
                    
                    if agent.phase.ID != agent.action.ID:
                        agent.state = np.asarray(agent.observe(self.eng, time, lanes_count, lane_vehs, veh_distance))
                        agent.update_wait_time(time, agent.action, agent.phase, lanes_count)
                        agent.set_phase(self.eng, agent.clearing_phase)
                        agent.action_freq = time + agent.clearing_time
                        agent.action_type = "update"
                       
                    else:
                        agent.action_freq = time + agent.green_time


                elif agent.action_type == "update":
                    agent.set_phase(self.eng, agent.action)
                    agent.action_freq = time + agent.green_time
                    agent.action_type = "reward"

        self.eng.next_step()

        
    def learning_step(self, time, done):
        """
        represents a single step of the simulation for the learning agent
        :param time: the current timestep
        :param done: flag indicating weather this has been the last step of the episode
        """
        lanes_count = self.eng.get_lane_vehicle_count()
        lane_vehs = self.eng.get_lane_vehicles()
        veh_distance = self.eng.get_vehicle_distance()
        waiting_vehs = None

        for agent in self.agents:
            if self.agents_type == "hybrid":
                agent.update_arr_dep_veh_num(lane_vehs)

            if time % agent.action_freq == 0:
                if agent.action_type == "reward":
                    reward = agent.get_reward(lanes_count)
                    reward = torch.tensor([reward], dtype=torch.float)
                    agent.reward = reward
                    agent.total_rewards += reward
                    agent.reward_count += 1
                    next_state = torch.FloatTensor(agent.observe(self.eng, time, lanes_count, lane_vehs, veh_distance)).unsqueeze(0)
                    self.memory.add(agent.state, agent.action.ID, reward, next_state, done)
                    agent.action_type = "act"
                                    
                if agent.action_type == "act":
                    agent.state = np.asarray(agent.observe(self.eng, time, lanes_count, lane_vehs, veh_distance))
                    agent.action = agent.act(self.local_net, agent.state, time, lanes_count, eps=self.eps)
                    # agent.update_clear_green_time(time)
                    # agent.green_time = max(5, int(np.max([agent.movements[x].green_time for x in agent.action.movements])))
                    agent.green_time = 10
                    
                    if agent.action.ID != agent.phase.ID:
                        agent.update_wait_time(time, agent.action, agent.phase, lanes_count)
                        agent.set_phase(self.eng, agent.clearing_phase)
                        agent.action_type = "update"
                        agent.action_freq = time + agent.clearing_time

                    else:
                        agent.action_type = "reward"
                        agent.action_freq = time + agent.green_time
                                                                
                elif agent.action_type == "update":
                    agent.set_phase(self.eng, agent.action)
                    agent.action_type = "reward"
                    agent.action_freq = time + agent.green_time


        if time % self.action_freq == 0: self.eps = max(self.eps-self.eps_decay,self.eps_end)
        # if time % self.eps_update == 0: self.eps = max(self.eps*self.eps_decay,self.eps_end)
        self.eng.next_step()

    def demand_step(self, time, done):
        """
        represents a single step of the simulation for the analytical agent
        :param time: the current timestep
        :param done: flag indicating weather this has been the last step of the episode, used for learning, here for interchangability of the two steps
        """
        print(time)
        lanes_count = self.eng.get_lane_vehicle_count()

        for agent in self.agents:
            if time % agent.action_freq == 0:
                if agent.action_type == "act":
                    agent.total_rewards += agent.get_reward(lanes_count)
                    agent.reward_count += 1
                    agent.action = agent.act(lanes_count)
                    agent.green_time = 10
                    
                    if agent.phase.ID != agent.action.ID:
                        agent.update_wait_time(time, agent.action, agent.phase, lanes_count)
                        agent.set_phase(self.eng, agent.clearing_phase)
                        agent.action_freq = time + agent.clearing_time
                        agent.action_type = "update"
                       
                    else:
                        agent.action_freq = time + agent.green_time

                elif agent.action_type == "update":
                    agent.set_phase(self.eng, agent.action)
                    agent.action_freq = time + agent.green_time
                    agent.action_type = "act"

        self.eng.next_step()


    # def policy_step(self, time, done):
    #     """
    #     represents a single step of the simulation for the policy gradient agent
    #     :param time: the current timestep
    #     :param done: flag indicating weather this has been the last step of the episode
    #     """
    #     lanes_count = self.eng.get_lane_vehicle_count()
    #     lane_vehs = self.eng.get_lane_vehicles()
    #     veh_distance = self.eng.get_vehicle_distance()
    #     waiting_vehs = None

    #     for agent in self.agents:
    #         if time % agent.action_freq == 0:
    #             if agent.action_type == "reward":
    #                 reward = agent.get_reward(lanes_count)
    #                 agent.ep_rews.append(reward)
    #                 agent.total_rewards += agent.get_reward(lanes_count)
    #                 agent.reward_count += 1
    #                 agent.action_type = "act"

    #             if agent.action_type == "act":

    #                 agent.state = np.asarray(agent.observe(self.eng, time, lanes_count, lane_vehs, veh_distance))
    #                 agent.batch_obs.append(agent.state.copy())
                    
    #                 # agent.action = agent.act(torch.as_tensor(agent.state, dtype=torch.float32), agent.logits_net)
    #                 agent.action = agent.act(torch.as_tensor(agent.state, dtype=torch.float32), self.logits_net)
    #                 agent.batch_acts.append(agent.action.ID)

    #                 agent.green_time = 10

    #                 # reward = agent.get_reward(lanes_count)
    #                 # agent.ep_rews.append(-reward)


    #                 if agent.phase.ID != agent.action.ID:
    #                     agent.update_wait_time(time, agent.action, agent.phase, lanes_count)
    #                     agent.set_phase(self.eng, agent.clearing_phase)
    #                     agent.action_freq = time + agent.clearing_time
    #                     agent.action_type = "update"
    #                 else:
    #                     agent.action_freq = time + agent.green_time
    #                     agent.action_type = "reward"

    #             elif agent.action_type == "update":
    #                agent.set_phase(self.eng, agent.action)
    #                agent.action_freq = time + agent.green_time
    #                agent.action_type = "reward"

    #         if done:
    #             # if episode is over, record info about episode               
                
    #             agent.ep_rews.append(0)

    #             if len(agent.ep_rews) > 1:
    #                 agent.batch_weights = agent.discount_cumsum(agent.ep_rews, agent.gamma)[:-1]
                    
    #                 with torch.no_grad():
    #                     # value_loss = agent.value_net(torch.from_numpy(np.asarray(agent.batch_obs)).float().unsqueeze(0))
    #                     value_loss = agent.value_net(torch.from_numpy(np.asarray(agent.batch_obs)).float().unsqueeze(0))

    #                 rews = np.asarray(agent.ep_rews[:-1]).reshape(np.asarray(agent.ep_rews[:-1]).shape[0], 1)
    #                 value_loss = value_loss.numpy()[0] + [0]

    #                 deltas = rews + agent.gamma * value_loss[1:] - value_loss[:-1]
    #                 adv = agent.discount_cumsum(deltas, agent.gamma * agent.lam)

    #                 # value_loss = agent.discount_cumsum(agent.ep_rews, 0.99) - value_loss.numpy()[0]
    #                 # value_loss = agent.ep_rews - value_loss.numpy()[0]
    #                 # value_loss = agent.discount_cumsum(agent.ep_rews - value_loss.numpy()[0], agent.gamma)

    #                 adv = (adv - adv.mean()) / adv.std()


    #                 self.policy_memory = [(agent.batch_acts[:-1], agent.batch_obs[:-1], agent.batch_weights, adv)]
                    
    #                 self.act_memory = [item for sublist in self.policy_memory for item in sublist[0]]
    #                 self.obs_memory = [item for sublist in self.policy_memory for item in sublist[1]]
    #                 self.weight_memory = [item for sublist in self.policy_memory for item in sublist[2]]
    #                 self.adv_memory = [item for sublist in self.policy_memory for item in sublist[3]]

    #                 self.memory = [(act, obs, weight, adv) for act, obs, weight, adv in zip(self.act_memory, self.obs_memory, self.weight_memory, self.adv_memory)]
    #                 random.shuffle(self.memory)
    #                 self.act_memory, self.obs_memory, self.weight_memory, self.adv_memory = zip(*self.memory)

    #                 # agent.pol_opt.zero_grad()
    #                 self.pol_opt.zero_grad()
    #                 batch_loss = agent.compute_loss(obs=torch.as_tensor(self.obs_memory, dtype=torch.float32),
    #                                       act=torch.as_tensor(self.act_memory, dtype=torch.int32),
    #                                       weights=torch.as_tensor(self.adv_memory, dtype=torch.float32),
    #                                       net=self.logits_net
    #                                       )
    #                 batch_loss.backward()
    #                 # agent.pol_opt.step()
    #                 self.pol_opt.step()

    #                 for t in range(1):                        
    #                     agent.val_opt.zero_grad()
    #                     value_loss = ((agent.value_net(torch.from_numpy(np.asarray(self.obs_memory)).float().unsqueeze(0)) -
    #                                    torch.from_numpy(np.asarray(self.weight_memory).copy()).float().unsqueeze(0))**2).mean()
    #                     value_loss.backward()
    #                     agent.val_opt.step()

    #                 agent.batch_obs = []          # for observations
    #                 agent.batch_acts = []         # for actions
    #                 agent.batch_weights = []      # for reward-to-go weighting in policy gradient
    #                 agent.ep_rews = []            # list for rewards accrued throughout ep


    #     self.eng.next_step()


    def reset(self):
        """
        resets the movements amd rewards for each agent and the simulation environment, should be called after each episode
        """
        self.eng.reset(seed=True)

        for agent in self.agents:
            agent.reset_movements()
            agent.action_freq = 10
            agent.total_rewards = 0
            agent.reward_count = 0
            agent.action_type = 'act'



