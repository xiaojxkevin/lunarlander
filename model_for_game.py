import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class QLearning(nn.Module):
    def __init__(self, maxlen=500000, batch_size=64, gamma=0.99, epsilon=1.0) -> None:
        super(QLearning, self).__init__()
        self.bs = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.linear1 = nn.Linear(7, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 5)
        # self.linear1 = nn.Linear(8, 256)
        # self.linear2 = nn.Linear(256, 128)
        # self.linear3 = nn.Linear(128, 4)
        self.memory_buffer = deque(maxlen=maxlen)
        self.rewards_list = []
        self.criterion = nn.MSELoss(reduction="sum")

    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        # assert False, "{}".format(x.shape)
        return x
    
    def choose_action(self, state:np.ndarray):
        if np.random.rand() < self.epsilon:
            return random.randint(0, 4)
        
        action_probs = F.softmax(self.forward(state), dim=1)
        action = torch.argmax(action_probs, dim=1)[0]
        return int(action)
    
    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory_buffer.append((state, action, reward, next_state, done))
    
    def get_random_samples_from_memory(self):
        random_sample = random.sample(self.memory_buffer, self.bs)
        return random_sample
    
    def get_attributes(self, random_sample):
        states = np.asarray([i[0] for i in random_sample])
        actions = torch.tensor([i[1] for i in random_sample])
        rewards = torch.tensor([i[2] for i in random_sample])
        next_states = np.asarray([i[3] for i in random_sample])
        done_list = torch.tensor([i[4] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return states, actions, rewards, next_states, done_list
    
    def compute_loss(self):
        # check memory size
        if len(self.memory_buffer) < self.bs:
            return None
        # Early stop if it is good enough
        if len(self.rewards_list) > 10 and np.mean(self.rewards_list[-10:]) >= 200:
            return None

        random_sample = self.get_random_samples_from_memory()
        # assert False, "\n{}\n{}".format(random_sample, len(random_sample))
        states, actions, rewards, next_states, done_list = self.get_attributes(random_sample)
        # assert False, "\n{}\n{}\n{}\n{}\n{}".format(states.shape, actions.shape, 
        #                                             rewards.shape, next_states.shape, done_list.shape)
        # tmp = self.forward(next_states)
        # assert False, "{}".format(tmp.shape)
        q_max, _ = torch.max(self.forward(next_states), dim=1)
        # assert False, "{}".format(q_max)
        targets = (rewards + self.gamma * q_max.detach() * (1 - done_list)).float()
        est_vec = self.forward(states)
        # target_vec = est_vec.detach().clone()
        # indexes = np.asarray([i for i in range(self.bs)], dtype=np.int32)
        est = est_vec[torch.arange(self.bs), actions]
        # target_vec[torch.arange(self.bs), actions] = targets.float()
        # assert False, "{}\n \nc{}".format(est.shape, 
        #                                 # target_vec, 
        #                                 targets.shape)
        loss = self.criterion(targets, est)
        return loss
        