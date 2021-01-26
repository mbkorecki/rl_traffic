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

    parser.add_argument("--sim_config", default='../scenarios/4x4/1.config',  type=str, help="the relative path to the simulation config file")

    parser.add_argument("--num_episodes", default=1, type=int,
                        help="the number of episodes to run (one episosde consists of a full simulation run for num_sim_steps)"
                        )
    parser.add_argument("--num_sim_steps", default=1800, type=int, help="the number of simulation steps, one step corresponds to 1 second")
    parser.add_argument("--agents_type", default='analytical', type=str, help="the type of agents learning/analytical")

    parser.add_argument("--update_freq", default=10, type=int,
                        help="the frequency of the updates (training pass) of the deep-q-network, default=10")
    parser.add_argument("--batch_size", default=64, type=int, help="the size of the mini-batch used to train the deep-q-network, default=64")
    parser.add_argument("--lr", default=5e-4, type=int, help="the learning rate for the dqn, default=5e-4")

    return parser.parse_args()

args = parse_args()

# eng = cityflow.Engine(args.sim_config, thread_num=8)
# for i in range(1800):
#     print(i)
#     eng.next_step()
# print(eng.get_average_travel_time(), eng.get_finished_vehicle_count())

logger = Logger(args)
environ = Environment(args, n_actions=9, n_states=33)

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
        environ.eng.set_replay_file(logger.log_path + "/replayFile.txt")
    
    print("episode ", i_episode)
    done = False

    environ.reset()

    t = 0
    while t < num_sim_steps:
        if t >= num_sim_steps-1: done = True
                            
        environ.step(t, done)   
        t += 1
      
        step = (step+1) % environ.update_freq
        if environ.agents_type == 'kearning' and step == 0:
            if len(environ.memory)>environ.batch_size:
                experience = environ.memory.sample()
                logger.losses.append(optimize_model(experience, environ.local_net, environ.target_net, environ.optimizer))

    logger.log_measures(environ)
    print(logger.reward, environ.eng.get_average_travel_time(), environ.eng.get_finished_vehicle_count())


logger.save_log_file(environ)
logger.serialise_data(environ)

if environ.agents_type == 'learning':
    logger.save_measures_plots()
    logger.save_models(environ)
