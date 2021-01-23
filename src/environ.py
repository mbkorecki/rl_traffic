import cityflow
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from dqn import DQN, ReplayMemory, optimize_model
from learning_agent import Learning_Agent
from analytical_agent import Analytical_Agent

class Environment:

    def __init__(self, args, n_actions=8, n_states=44):

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
        self.num_phases = []
        if args.agents_type == 'analytical':
            self.step = self.analytical_step
        else:
            self.step = self.learning_step
        
        agent_ids = [x for x in self.eng.get_intersection_ids() if not self.eng.is_intersection_virtual(x)]
        for agent_id in agent_ids:
            if args.agents_type == 'analytical':
                new_agent = Analytical_Agent(self.eng, ID=agent_id)
            else:
                new_agent = Learning_Agent(self.eng, ID=agent_id, in_roads=self.eng.get_intersection_in_roads(agent_id), out_roads=self.eng.get_intersection_out_roads(agent_id))

            self.num_phases.append(len(new_agent.phases))
            non_clearing_phases = [x for x in new_agent.phases.keys() if new_agent.phases[x].movements != []]
            if len(new_agent.phases) <= 1 and new_agent.ID != '3630249566':
                new_agent.set_phase(self.eng, list(new_agent.phases.keys())[0])
            else:
                if len(new_agent.phases) == 2:
                    self.special_agents.append(new_agent)
                else:
                    self.agents.append(new_agent)

       
        print(len(self.num_phases))
        print(np.histogram(self.num_phases, bins = range(max(self.num_phases))))

        print(len(self.agents))
        print(len(self.special_agents))
        self.action_freq = 10   #typical update freq for agents

    def analytical_step(self, time, done, log_phases):
        print(time)
        lane_vehs = self.eng.get_lane_vehicles()
        waiting_vehs = self.eng.get_lane_waiting_vehicle_count()
        
        for agent in self.special_agents:
            if time % agent.action_freq == 0:
                if agent.action_type == "act":
                    agent.set_phase(self.eng, agent.clearing_phase)
                    agent.action_freq = time + 5
                    agent.action_type = "update"
                else:
                    phase = [x for x in agent.phases.keys() if agent.phases[x].movements != []][0]
                    agent.set_phase(self.eng, phase)
                    agent.action_freq = time + 30
                    agent.action_type = "act"
                    
        for agent in self.agents:
            if time % agent.action_freq == 0:
                if agent.action_type == "act":
                    agent.update_arr_dep_veh_num(lane_vehs)
                    agent.action, agent.green_time = agent.act(self.eng, time, waiting_vehs)
                    if agent.phase.ID != agent.action.ID:
                        agent.set_phase(self.eng, agent.clearing_phase)
                        agent.action_freq = time + 2
                        agent.action_type = "update"

                        ###LOGGING DATA###
                        # if log_phases:
                        #     agent.total_duration[agent.phase+1].append(agent.phases_duration[agent.phase+1])
                        #     agent.phases_duration[agent.phase+1] = 0
                        
                    else:
                        # agent.action_freq = time + self.action_freq
                        agent.action_freq = time + agent.green_time


                elif agent.action_type == "update":
                    agent.update_wait_time(agent.action, agent.green_time, waiting_vehs)
                    agent.set_phase(self.eng, agent.action)
                    agent.action_freq = time + agent.green_time
                    agent.action_type = "act"
                    
                    ###LOGGING DATA###
                    # if log_phases:
                    #     agent.total_duration[agent.phase+1].append(agent.phases_duration[agent.phase+1])
                    #     agent.phases_duration[agent.phase+1] = 0


            ##### LOGGING DATA ######
            # if log_phases:
            #     agent.phases_duration[agent.phase] += 1
            #     agent.past_phases.append(agent.phase)

        self.eng.next_step()

        
    def learning_step(self, time, done, log_phases):
        lanes_count = None
        waiting_vehs = None

        for agent in self.agents:
            # agent.update_arr_dep_veh_num(self.eng)
            if time % agent.action_freq == 0:
                if agent.action_type == "reward":
                    if lanes_count == None: lanes_count = self.eng.get_lane_vehicle_count()
                    reward = agent.get_reward(self.eng, time, lanes_count)
                    reward = torch.tensor([reward], dtype=torch.float)
                    agent.reward = reward
                    # agent.total_rewards += reward
                    # agent.reward_count += 1
                    next_state = torch.FloatTensor(agent.observe(self.eng, time, lanes_count)).unsqueeze(0)
                    self.memory.add(agent.state, agent.action.ID, reward, next_state, done)
                    agent.action_type = "act"
                                    
                if agent.action_type == "act":
                    if lanes_count == None: lanes_count = self.eng.get_lane_vehicle_count()
                    agent.state = np.asarray(agent.observe(self.eng, time, lanes_count))
                    agent.action = agent.act(self.local_net, agent.state, eps=self.eps)
                    agent.green_time = 10
   
                    if agent.action != agent.phase:
                        agent.set_phase(self.eng, agent.clearing_phase)
                        agent.action_type = "update"
                        agent.action_freq = time + 2

                                                    
                        ##### LOGGING DATA ######
                        # if log_phases:
                        #     agent.total_duration[agent.phase+1].append(agent.phases_duration[agent.phase+1])
                        #     agent.phases_duration[agent.phase+1] = 0
                        
                    else:
                        agent.action_type = "reward"
                        agent.action_freq = time + agent.green_time
                                                                
                elif agent.action_type == "update":
                    if waiting_vehs == None: waiting_vehs = self.eng.get_lane_waiting_vehicle_count()
                    agent.update_wait_time(agent.action, agent.green_time, waiting_vehs)
                    agent.set_phase(self.eng, agent.action)
                    agent.action_type = "reward"
                    agent.action_freq = time + agent.green_time
                
                    ##### LOGGING DATA ######
                    # if log_phases:
                    #     agent.total_duration[agent.phase+1].append(agent.phases_duration[agent.phase+1])
                    #     agent.phases_duration[agent.phase+1] = 0
                       

            ##### LOGGING DATA ######
            # if log_phases:
            #     agent.phases_duration[agent.phase] += 1
            #     agent.past_phases.append(agent.phase)

        if time % self.action_freq == 0: self.eps = max(self.eps-self.eps_decay,self.eps_end)
        self.eng.next_step()

    def reset(self):
        self.eng.reset(seed=False)

        for agent in self.agents:
            agent.reset_movements()
