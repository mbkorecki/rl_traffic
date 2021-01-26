import cityflow
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from dqn import DQN, ReplayMemory, optimize_model
from learning_agent import Learning_Agent
from analytical_agent import Analytical_Agent
from demand_agent import Demand_Agent

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

        self.n_actions = n_actions
        self.n_states = n_states
        
        self.local_net = DQN(n_states, n_actions, seed=2).to(self.device)
        self.target_net = DQN(n_states, n_actions, seed=2).to(self.device)

        self.update_freq = args.update_freq      # how often to update the network
        self.batch_size = args.batch_size
        
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps_decay= 5e-5
        self.eps = self.eps_start
        
        self.optimizer = optim.Adam(self.local_net.parameters(), lr=args.lr, amsgrad=True)
        self.memory = ReplayMemory(self.n_actions, batch_size=args.batch_size)

        self.agents = []
        self.special_agents = []

        self.agents_type = args.agents_type
        if args.agents_type == 'analytical':
            self.step = self.analytical_step
        elif args.agents_type == 'learning':
            self.step = self.learning_step
        elif args.agents_type == 'demand':
            self.step = self.demand_step
        else:
            raise Exception("The specified agent type:", args.agents_type, "is incorrect, choose from: analytical/learning/demand")  
        
        agent_ids = [x for x in self.eng.get_intersection_ids() if not self.eng.is_intersection_virtual(x)]
        for agent_id in agent_ids:
            if args.agents_type == 'analytical':
                new_agent = Analytical_Agent(self.eng, ID=agent_id)
            elif args.agents_type == 'learning':
                new_agent = Learning_Agent(self.eng, ID=agent_id, in_roads=self.eng.get_intersection_in_roads(agent_id), out_roads=self.eng.get_intersection_out_roads(agent_id))
            elif args.agents_type == 'demand':
                new_agent = Demand_Agent(self.eng, ID=agent_id)
            else:
                raise Exception("The specified agent type:", args.agents_type, "is incorrect, choose from: analytical/learning/demand")  


            if len(new_agent.phases) <= 1:
                new_agent.set_phase(self.eng, new_agent.phases[0])
            else:
                self.agents.append(new_agent)

        self.action_freq = 10   #typical update freq for agents
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
        waiting_vehs = None
        
        for agent in self.agents:
            agent.update_arr_dep_veh_num(lane_vehs)
            if time % agent.action_freq == 0:
                if agent.action_type == "act":
                    agent.total_rewards += agent.get_reward(lanes_count)
                    agent.reward_count += 1
                    agent.action, agent.green_time = agent.act(self.eng, time)
                    
                    if agent.phase.ID != agent.action.ID:
                        agent.set_phase(self.eng, agent.clearing_phase)
                        agent.action_freq = time + agent.clearing_time
                        agent.action_type = "update"
                       
                    else:
                        agent.action_freq = time + agent.green_time


                elif agent.action_type == "update":
                    if waiting_vehs == None: waiting_vehs = self.eng.get_lane_waiting_vehicle_count()
                    agent.update_wait_time(agent.action, agent.green_time, waiting_vehs)
                    agent.set_phase(self.eng, agent.action)
                    agent.action_freq = time + agent.green_time
                    agent.action_type = "act"

        self.eng.next_step()

        
    def learning_step(self, time, done):
        """
        represents a single step of the simulation for the learning agent
        :param time: the current timestep
        :param done: flag indicating weather this has been the last step of the episode
        """
        lanes_count = self.eng.get_lane_vehicle_count()
        lane_vehs = self.eng.get_lane_vehicles()
        waiting_vehs = None

        for agent in self.agents:
            agent.update_arr_dep_veh_num(lane_vehs)
            if time % agent.action_freq == 0:
                if agent.action_type == "reward":
                    reward = agent.get_reward(lanes_count)
                    reward = torch.tensor([reward], dtype=torch.float)
                    agent.reward = reward
                    agent.total_rewards += reward
                    agent.reward_count += 1
                    next_state = torch.FloatTensor(agent.observe(self.eng, time, lanes_count)).unsqueeze(0)
                    self.memory.add(agent.state, agent.action.ID, reward, next_state, done)
                    agent.action_type = "act"
                                    
                if agent.action_type == "act":
                    agent.state = np.asarray(agent.observe(self.eng, time, lanes_count))
                    agent.action = agent.act(self.local_net, agent.state, time, eps=self.eps)
                    agent.green_time = 10
                    
                    if agent.action != agent.phase:
                        agent.set_phase(self.eng, agent.clearing_phase)
                        agent.action_type = "update"
                        agent.action_freq = time + agent.clearing_time

                    else:
                        agent.action_type = "reward"
                        agent.action_freq = time + agent.green_time
                                                                
                elif agent.action_type == "update":
                    if waiting_vehs == None: waiting_vehs = self.eng.get_lane_waiting_vehicle_count()
                    agent.update_wait_time(agent.action, agent.green_time, waiting_vehs)
                    agent.set_phase(self.eng, agent.action)
                    agent.action_type = "reward"
                    agent.action_freq = time + agent.green_time


        if time % self.action_freq == 0: self.eps = max(self.eps-self.eps_decay,self.eps_end)
        self.eng.next_step()

    def demand_step(self, time, done):
        """
        represents a single step of the simulation for the analytical agent
        :param time: the current timestep
        :param done: flag indicating weather this has been the last step of the episode, used for learning, here for interchangability of the two steps
        """
        print(time)
        lanes_count = self.eng.get_lane_vehicle_count()
        lanes_count = self.eng.get_lane_vehicle_count()

        for agent in self.agents:
            if time % agent.action_freq == 0:
                if agent.action_type == "act":
                    agent.total_rewards += agent.get_reward(lanes_count)
                    agent.reward_count += 1
                    agent.action = agent.act(lanes_count)
                    agent.green_time = 10
                    
                    if agent.phase.ID != agent.action.ID:
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
        
    def reset(self):
        """
        resets the movements amd rewards for each agent and the simulation environment, should be called after each episode
        """
        self.eng.reset(seed=False)

        for agent in self.agents:
            agent.reset_movements()
            agent.total_rewards = 0
            agent.reward_count = 0

