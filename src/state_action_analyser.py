import argparse
import os
import random
import dill

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--memory_file", default='../4x4mount_config1_analytical/memory.dill',  type=str, help="the relative path to the memory file")

    return parser.parse_args()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

args = parse_args()

with open(args.memory_file, "rb") as f:
    memory = dill.load(f)

actions = []
states = []
rewards = []

state_action_dict = {}
state_reward_dict = {}

for elem in memory:
    actions.append(elem.action)
    states.append(elem.state)
    rewards.append(elem.reward)
    if elem.action not in state_action_dict.keys():
        state_action_dict.update({elem.action : [elem.state]})
    else:
        state_action_dict[elem.action].append(elem.state)
        
pca_state_action_dict = {}
pca = PCA(n_components=2)

fig = plt.figure()
# ax = plt.axes(projection='3d')
ax = plt.axes()
print(len(states))

i = 0
cmap = get_cmap(9)
for action in state_action_dict.keys():    
    state_action_dict.update({action : pca.fit_transform(state_action_dict[action])})
    x,y = zip(*state_action_dict[action])

    ax.scatter(x, y, s=2, color=cmap(i))
    i += 1
        
# plt.hist(actions, bins=100)

plt.xlim([-1.5, 2])
plt.ylim([-1, 2])

plt.show()


# min_reward = min(rewards)
# max_reward = max(rewards)

# bins = 10

# increment = (min_reward - max_reward) / bins

# edge_values = []
# for i in range(bins):
#     edge_values.append((i+1) * increment)
# edge_values = list(reversed(edge_values))
# edge_values = edge_values[:-1]
# edge_values.append(0)
# cat_rewards = []

# for reward in rewards:
#     for i, val in enumerate(edge_values):
#         if reward <= val:
#             cat_rewards.append(i)
#             break

# for state, reward in zip(states, cat_rewards):
#     if reward not in state_reward_dict.keys():
#         state_reward_dict.update({reward : [state]})
#     else:
#         state_reward_dict[reward].append(state)

# cmap = get_cmap(10)
# i = 0
# for reward in state_reward_dict.keys():
#     if len(state_reward_dict[reward]) > 2:
#         state_reward_dict.update({reward : pca.fit_transform(state_reward_dict[reward])})
#         x,y = zip(*state_reward_dict[reward])

#         plt.scatter(x, y, s=2, color=cmap(i))
#     i += 1
# plt.show()
