import numpy as np
import random 
from collections import namedtuple, deque 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
GAMMA = 0.999           # discount factor
TAU = 1e-3              # for soft update of target parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed=2, fc1_unit=128,
                 fc2_unit = 64):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(DQN,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1= nn.Linear(state_size,fc1_unit)
        self.fc2 = nn.Linear(fc1_unit,fc2_unit)
        self.fc3 = nn.Linear(fc2_unit,action_size)
    
    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def optimize_model(experiences, net_local, net_target, optimizer, gamma=GAMMA):
    """Update value parameters using given batch of experience tuples.

    Params
    =======

    experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
    
    gamma (float): discount factor
    """
    states, actions, rewards, next_state, dones = experiences
    criterion = torch.nn.MSELoss()
    net_local.train()
    net_target.eval()
    
    #shape of output from the model (batch_size,action_dim) = (64,4)
    predicted_targets = net_local(states).gather(1,actions)
    
    with torch.no_grad():
        labels_next = net_target(next_state).detach().max(1)[0].unsqueeze(1)

    # .detach() ->  Returns a new Tensor, detached from the current graph.
    labels = rewards + (gamma * labels_next * (1-dones))

    loss = criterion(predicted_targets,labels).to(device)
    optimizer.zero_grad()
    loss.backward()
    
    for param in net_local.parameters():
        param.grad.data.clamp_(-1, 1)
    # torch.nn.utils.clip_grad.clip_grad_norm_(net_local.parameters(), 10)

    optimizer.step()

    # ------------------- update target network ------------------- #
    soft_update(net_local, net_target,TAU)
    return loss.item()

        
def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    =======
        local model (PyTorch model): weights will be copied from
        target model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
        
    """
    for target_param, local_param in zip(target_model.parameters(),
                                         local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)


class ReplayMemory:
    """Fixed -size buffe to store experience tuples."""
    
    def __init__(self, action_size, buffer_size=BUFFER_SIZE, batch_size=64, seed=2):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])
        self.seed = random.seed(seed)
        
    def add(self,state, action, reward, next_state,done):
        """Add a new experience to memory."""
        e = self.experiences(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory,k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
