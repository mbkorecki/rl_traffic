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

def draw_pca_analysis(ax, path):
    with open(path, "rb") as f:
        memory = dill.load(f)

    actions = []
    states = []
    rewards = []

    state_action_dict = {}
    state_reward_dict = {}

    # for elem in memory:
    #     actions.append(elem.action)
    #     states.append(elem.state)
    #     rewards.append(elem.reward)
    #     if elem.action not in state_action_dict.keys():
    #         state_action_dict.update({elem.action : [elem.state]})
    #     else:
    #         state_action_dict[elem.action].append(elem.state)


    for elem in memory:
        actions.append(elem[1].ID)
        states.append(elem[0])
        if elem[1].ID not in state_action_dict.keys():
            state_action_dict.update({elem[1].ID : [elem[0]]})
        else:
            state_action_dict[elem[1].ID].append(elem[0])

    
    pca_state_action_dict = {}
    pca = PCA(n_components=2)

    # ax = plt.axes()
    print(len(states))

    i = 0

    all_states = [state_action_dict[x] for x in state_action_dict.keys()]
    all_states = [x for sub in all_states for x in sub]

    pca.fit(all_states)
    print(pca.explained_variance_ratio_)

    for action in range(10):
        # print(action, len(state_action_dict[action]))
        if action in state_action_dict.keys():
            state_action_dict.update({action : pca.transform(state_action_dict[action])})
            # state_action_dict.update({action : pca.fit_transform(state_action_dict[action])})

            x,y = zip(*state_action_dict[action])

            ax.scatter(x, y, s=10, color=cmap(action), alpha=0.5, label='Action ' + str(action))
        
            ax.set_xlabel('Projection PC1 ' + "({:.2f})".format(pca.explained_variance_ratio_[0]))
            ax.set_ylabel('Projection PC2 ' + "({:.2f})".format(pca.explained_variance_ratio_[1]))
        


args = parse_args()
paths = ['../state_action/ny196_config1_analytical/agent_history.dill',
         '../state_action/ny196_config1_hybrid_load(1)/agent_history.dill',
         '../state_action/ny196_config1_presslight_load(1)/agent_history.dill']
names = ['Analytic', 'GuidedLight', 'PressLight']
fig, axes = plt.subplots(1,len(paths), constrained_layout=True, sharex=True, sharey=True)

cmap = get_cmap(10)


# for (j, path), name in zip(enumerate(paths), names):
#     draw_pca_analysis(axes[0, j], path)
#     axes[0,j].set_title(name)
#     if j == 0 or j == 1:
#         axes[0,j].set_xlim([-1, 1.5])
#         axes[0,j].set_ylim([-1.1, 1.5])

# # fig.supxlabel('Projection PC1')
# # fig.supylabel('Projection PC2')
# axes[0,2].legend(loc="upper center", prop={'size': 6})
# plt.show()


def draw_histogram(ax, path):
    with open(path, "rb") as f:
        memory = dill.load(f)
    actions = []
    for elem in memory:
        actions.append(elem[1].ID)

    _, _, patches = ax.hist(actions, bins=np.arange(0,10)-0.5)
    ax.set_xticks(range(0,9))
    for i, patch in enumerate(patches):
        patch.set_facecolor(cmap(i))


for (j, path), name in zip(enumerate(paths), names):
    draw_histogram(axes[j], path)
    axes[j].set_title(name)

axes[1].set_xlabel('Actions')

plt.show()
