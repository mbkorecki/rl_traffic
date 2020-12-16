import cityflow
import torch
import torch.optim as optim
import numpy as np

from dqn import DQN, ReplayMemory, optimize_model
from intersection import Intersection

class Environment:

    def __init__(self, args, n_actions=8, n_states=44):

        self.eng = cityflow.Engine(args.sim_config, thread_num=8)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_actions = n_actions
        self.n_states = n_states
        
        self.local_net = DQN(n_states, n_actions, seed=2).to(self.device)
        self.target_net = DQN(n_states, n_actions, seed=2).to(self.device)

        self.update_freq = args.update_freq      # how often to update the network
        self.action_freq = 10   #typical update freq for agents
        self.batch_size = args.batch_size
        
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay= 5e-5
        self.eps = self.eps_start
        
        self.optimizer = optim.Adam(self.local_net.parameters(), lr=args.lr, amsgrad=True)
        self.memory = ReplayMemory(self.n_actions, batch_size=args.batch_size)

        self.agents = []

        agent_ids = [x for x in self.eng.get_intersection_ids() if not self.eng.is_intersection_virtual(x)]
        for agent_id in agent_ids:
            newAgent = Intersection(phase=0, ID=agent_id,
                                    in_roads=self.eng.get_intersection_in_roads(agent_id),
                                    out_roads=self.eng.get_intersection_out_roads(agent_id)
                                    )
            newAgent.init_movements_lanes_dict(self.eng)
            
            self.agents.append(newAgent)


        self.phase_to_vec = self.agents[0].phase_to_vec
        self.phase_to_movement = self.agents[0].phase_to_movement
        self.movement_to_phase = self.agents[0].movement_to_phase
        

    def step(self, time, done, log_phases):

        for agent in self.agents:
            if time % agent.action_freq == 0:
                if agent.action_type == "reward":

                    reward = agent.get_reward(self.eng)
                    reward = torch.tensor([reward], dtype=torch.float)
                    agent.reward = reward
                    agent.total_rewards += reward

                    next_state = torch.FloatTensor(agent.observe(self.eng)).unsqueeze(0)
                    self.memory.add(agent.state, agent.action, reward, next_state, done)

                    agent.action_type = "act"
                    agent.action_freq = self.action_freq
                                    
                if agent.action_type == "act":
                    state = np.asarray(agent.observe(self.eng))
                    action = agent.act(self.local_net, state, eps=self.eps)

                    #ADD CHECK IF THERE IS A CAR ON THAT LANE
                    movements = self.phase_to_movement[agent.phase]
                    for i in range(12):
                        if i not in movements:
                            agent.current_wait_time[i] += 10
                        
                    if action != agent.phase:
                        movements = self.phase_to_movement[action]
                        for move in movements:
                            if agent.max_wait_time[move] < agent.current_wait_time[move]:
                                agent.max_wait_time[move] = agent.current_wait_time[move]
                        
                            agent.current_wait_time[move] = 0

                        agent.set_phase(self.eng, -1)
                        agent.action_type = "update"
                        agent.action_freq = time+2

                                                    
                        ##### LOGGING DATA ######
                        if log_phases:
                            agent.total_duration[agent.phase+1].append(agent.phases_duration[agent.phase+1])
                            agent.phases_duration[agent.phase+1] = 0
                        
                    else:
                        agent.action_type = "reward"
                        agent.action_freq = self.action_freq
                            
                    agent.state = state
                    agent.action = action
                                    
                elif agent.action_type == "update":
                        
                    agent.set_phase(self.eng, agent.action)
                    agent.action_type = "reward"
                    agent.action_freq = self.action_freq
                
                    ##### LOGGING DATA ######
                    if log_phases:
                        agent.total_duration[agent.phase+1].append(agent.phases_duration[agent.phase+1])
                        agent.phases_duration[agent.phase+1] = 0
                       

            ##### LOGGING DATA ######
            if log_phases:
                agent.phases_duration[agent.phase] += 1
                agent.past_phases.append(agent.phase)

        if time % self.action_freq == 0: self.eps = max(self.eps-self.eps_decay,self.eps_end)
        self.eng.next_step()

    def reset(self):
        self.eng.reset(seed=False)

        for agent in self.agents:
            agent.reset_veh_num()
            agent.max_wait_time = np.zeros(12)
            agent.current_wait_time = np.zeros(12)
