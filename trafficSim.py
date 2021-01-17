import cityflow
import numpy as np

import math
import random

from itertools import count
import torch
import torch.optim as optim

from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F
import random

import argparse
import os

from dqn import DQN, ReplayMemory, optimize_model
from environ import Environment
from logger import Logger

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sim_config", default='4x4/1.config',  type=str, help="the relative path to the simulation config file")

    parser.add_argument("--num_episodes", default=150, type=int,
                        help="the number of episodes to run (one episosde consists of a full simulation run for num_sim_steps)"
                        )
    parser.add_argument("--num_sim_steps", default=1800, type=int, help="the number of simulation steps, one step corresponds to 1 second")

    parser.add_argument("--update_freq", default=10, type=int,
                        help="the frequency of the updates (training pass) of the deep-q-network, default=50")
    parser.add_argument("--batch_size", default=64, type=int, help="the size of the mini-batch used to train the deep-q-network, default=64")
    parser.add_argument("--lr", default=5e-4, type=int, help="the learning rate for the dqn, default=5e-4")
    parser.add_argument("--agents_type", default='analysis', type=str, help="the type of agents")

    return parser.parse_args()

args = parse_args()

logger = Logger(args)
environ = Environment(args, n_actions=8, n_states=32)

num_episodes = args.num_episodes
num_sim_steps = args.num_sim_steps

step = 0

environ.eng.set_save_replay(open=False)
environ.eng.set_random_seed(2)

log_phases = False

for i_episode in range(num_episodes):
    logger.losses = []
    if i_episode == num_episodes-1:
        environ.eng.set_save_replay(open=True)
        environ.eng.set_replay_file("../" + logger.log_path + "/replayFile.txt")
        log_phases = True
    
    print("episode ", i_episode)
    done = False

    environ.reset()

    t = 0
    while t < num_sim_steps:
        if t >= num_sim_steps-1: done = True
                            
        environ.step(t, done, log_phases)   
        t += 1
      
        step = (step+1) % environ.update_freq
        if step == 0:
            if len(environ.memory)>environ.batch_size:
                experience = environ.memory.sample()
                logger.losses.append(optimize_model(experience, environ.local_net, environ.target_net, environ.optimizer))

    logger.log_measures(environ)
    print(logger.reward, environ.eng.get_average_travel_time(), environ.eng.get_finished_vehicle_count())


# for agent in environ.agents:
#     for i in range(12):
#         if agent.max_wait_time[agent.phase] < agent.current_wait_time[agent.phase]:
#             agent.max_wait_time[agent.phase] = agent.current_wait_time[agent.phase]


# logger.save_log_file(environ)
# logger.save_phase_plots(environ)
# logger.save_measures_plots()
# logger.save_models(environ)



# def read_light_file(path):
#     light_file = open(path, 'r')
#     light_file = light_file.read().splitlines()
#     time_phase_list = []
    
#     for line in light_file:
#         time, phase = line.split(",")
#         time_phase_list.append((int(float(time)),int(float(phase))))

#     return time_phase_list

# eng = cityflow.Engine(args.sim_config, thread_num=8)
# agent_phases_dict = {}
# agent_idx = {}
# agent_ids = [x for x in eng.get_intersection_ids() if not eng.is_intersection_virtual(x)]

# for agent_id in agent_ids:
#     path = 'hangzhou/lights/signal_inter_' + agent_id + '.txt'
#     time_phase_list = read_light_file(path)
#     agent_phases_dict.update({agent_id : time_phase_list})
#     agent_idx.update({agent_id : 0})
#     eng.set_tl_phase(agent_id, 0)

    
# for t in range(600):
#     eng.next_step()

#     # for agent_id in agent_ids:
#     #     if len(agent_phases_dict[agent_id]) <  agent_idx[agent_id]:
#     #         schedule = agent_phases_dict[agent_id][agent_idx[agent_id]]
#     #         if schedule[0] == t:
#     #             eng.set_tl_phase(agent_id, schedule[1])
#     #             agent_idx[agent_id] += 1

#     print(t)
# print(eng.get_average_travel_time(), eng.get_finished_vehicle_count())
