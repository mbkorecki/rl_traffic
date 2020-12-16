import cityflow
import numpy as np

import math
import random
import matplotlib
import matplotlib.pyplot as plt
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
from intersection import Intersection
from environ import Environment

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sim_config", default='4x4/cityflow.config',  type=str, help="the relative path to the simulation config file")

    parser.add_argument("--num_episodes", default=150, type=int,
                        help="the number of episodes to run (one episosde consists of a full simulation run for num_sim_steps)"
                        )
    parser.add_argument("--num_sim_steps", default=1800, type=int, help="the number of simulation steps, one step corresponds to 1 second")

    parser.add_argument("--update_freq", default=50, type=int,
                        help="the frequency of the updates (training pass) of the deep-q-network, default=50")
    parser.add_argument("--batch_size", default=64, type=int, help="the size of the mini-batch used to train the deep-q-network, default=64")
    parser.add_argument("--lr", default=5e-4, type=int, help="the learning rate for the dqn, default=5e-4")

    return parser.parse_args()

args = parse_args()

log_path = "results" + args.sim_config.split('/')[0] + '-' + str(args.num_episodes) + '-' + str(args.update_freq)
old_path = log_path
i = 1

while os.path.exists(log_path):
    log_path = old_path + "(" + str(i) + ")"
    i += 1

os.mkdir(log_path)


environ = Environment(args, n_actions=8, n_states=44)

num_episodes = args.num_episodes
num_sim_steps = args.num_sim_steps

veh_count = []
travel_time = []
losses = []
plot_rewards = []
episode_losses = []

eps = environ.eps_start
step = 0

environ.eng.set_save_replay(open=False)
environ.eng.set_random_seed(2)

log_phases = False

for i_episode in range(num_episodes):
    losses = []
    if i_episode == num_episodes-1:
        environ.eng.set_save_replay(open=True)
        environ.eng.set_replay_file("../" + log_path + "/replayFile.txt")
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
                losses.append(optimize_model(experience, environ.local_net, environ.target_net, environ.optimizer))
                
    reward = 0
    for agent in environ.agents:
        reward += (agent.total_rewards / (num_sim_steps / agent.action_freq))
        agent.total_rewards = 0
        
    print(reward, environ.eng.get_average_travel_time(), environ.eng.get_finished_vehicle_count())
    plot_rewards.append(reward)
    veh_count.append(environ.eng.get_finished_vehicle_count())
    travel_time.append(environ.eng.get_average_travel_time())
    episode_losses.append(np.mean(losses))


for agent in environ.agents:
    for i in range(12):
        if agent.max_wait_time[agent.phase] < agent.current_wait_time[agent.phase]:
            agent.max_wait_time[agent.phase] = agent.current_wait_time[agent.phase]


 
log_file = open(log_path + "/logs.txt","w+")

log_file.write(str(args.sim_config))
log_file.write("\n")
log_file.write(str(args.num_episodes))
log_file.write("\n")
log_file.write(str(args.num_sim_steps))
log_file.write("\n")
log_file.write(str(args.update_freq))
log_file.write("\n")
log_file.write(str(args.batch_size))
log_file.write("\n")
log_file.write(str(args.lr))
log_file.write("\n")

log_file.write("mean vehicle count: " + str(np.mean(veh_count[num_episodes-10:])) + " with sd: " + str(np.std(veh_count[num_episodes-10:])) +
      "\nmean travel time: " + str(np.mean(travel_time[num_episodes-10:])) + " with sd: " + str(np.std(travel_time[num_episodes-10:])))
log_file.write("\n")
log_file.write("\n")

for agent in environ.agents:
    log_file.write(agent.ID + "\n")
    for i in range(-1, environ.n_actions):
        log_file.write("phase " + str(i) + " duration: " + str(agent.past_phases.count(i)) + "\n")
    log_file.write("\n")

    for i in range(-1, environ.n_actions):
        log_file.write("phase " + str(i) + " switch: " + str(len(agent.total_duration[i+1])) + "\n")
    log_file.write("\n")

    for i in range(12):
        log_file.write("movement " + str(i) + " max wait time: " + str(agent.max_wait_time[i]) + "\n")
    log_file.write("\n")
    
log_file.write("\n")
        
log_file.close()

for agent in environ.agents:
    plt.plot(agent.past_phases, '|', linewidth=25)
    figure = plt.gcf()
    figure.set_size_inches(20,10)
    plt.xticks(np.arange(0, num_sim_steps+1, step=10))
    plt.ylabel('phase')
    plt.xlabel('time')
    plt.grid()
    ax = plt.gcf().get_axes()[0]
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(log_path + '/phase' + str(agent.ID) + '.png', bbox_inches='tight')
    plt.clf()
    
plt.plot(veh_count)
plt.ylabel('vehicle count')
plt.xlabel('episodes')
plt.savefig(log_path + '/vehCount.png')
plt.clf()

plt.plot(travel_time)
plt.ylabel('avg travel time')
plt.xlabel('episodes')
plt.savefig(log_path + '/avgTime.png')
plt.clf()

plt.plot(plot_rewards)
plt.ylabel('total rewards')
plt.xlabel('episodes')
plt.savefig(log_path + '/totalRewards.png')
plt.clf()

plt.plot(episode_losses)
plt.ylabel('q loss')
plt.xlabel('episodes')
plt.savefig(log_path + '/qLosses.png')
plt.clf()

torch.save(environ.local_net.state_dict(), log_path + '/policy_net.pt')
torch.save(environ.target_net.state_dict(), log_path + '/target_net.pt')

