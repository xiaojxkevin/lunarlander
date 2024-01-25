import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
class QLearning(nn.Module):
    def __init__(self, maxlen=500000, batch_size=64, gamma=0.99, epsilon=1.0) -> None:
        super(QLearning, self).__init__()
        self.bs = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.linear1 = nn.Linear(8, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 4)
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
            return random.randint(0, 3)
        
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
        


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.encoder = nn.Linear(8, 128)

        self.action_layer = nn.Linear(128, 4)
        self.value_layer = nn.Linear(128, 1)

        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, x):
        x = torch.from_numpy(x).float()
        # Endocding
        x = F.relu(self.encoder(x))

        # Decoding
        state_value = self.value_layer(x)

        action_probs = F.softmax(self.action_layer(x), dim=0)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)

        return action.item()

    def calculateLoss(self, gamma=0.99):

        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())

        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)
        return loss

    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]


class OfflineClassifier(nn.Module):
    def __init__(self) -> None:
        super(OfflineClassifier, self).__init__()
        self.linear1 = nn.Linear(8, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        return x



class Bayes(nn.Module):
    def __init__(self):
        super(Bayes, self).__init__()
        # self.evi = nn.Sequential(
        #   nn.Linear(9, 128),
        #   nn.ReLU(),
        #   nn.Linear(128, 128),
        #   nn.ReLU(),
        #     nn.Linear(128, 1),
        #     nn.Sigmoid
        # )
        self.cond = nn.Sequential(
          nn.Linear(9, 128),
          nn.ReLU(),
          nn.Linear(128, 128),
          nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        self.reward_li=[]
        self.prob = []
    def forward(self, x2):
        # evi_i = torch.from_numpy(x1).float()
        cond_i=torch.from_numpy(x2).float()
        cond_o=F.softmax(self.cond(cond_i), dim=0)
        action_distribution = Categorical(cond_o)
        action = action_distribution.sample()
        self.prob.append(cond_o)
        return action.item()

    def calculateLoss(self,gamma):
        dis_reward = 0
        target=300
        temp_reward=0
        # self.reward_li=torch.tensor(self.reward_li)
        for i in range(len(self.prob)):
            temp_reward+=self.prob[i][0]*self.reward_li[i]
            temp_reward+=self.prob[i][1]*self.reward_li[i]
            temp_reward+=self.prob[i][2]*self.reward_li[i]
            temp_reward+=self.prob[i][3]*self.reward_li[i]
            dis_reward=temp_reward+gamma*dis_reward
            temp_reward=0
            # rewards.insert(0, dis_reward)
        
        # normalizing the rewards:
        target=torch.tensor(target).float()
        loss=F.mse_loss(target, dis_reward)
        return loss
    def clearMemory(self):
        del self.reward_li[:]
        del self.prob[:]


class MixGaussian(object):
    def __init__(self,sigma_1,sigma_2,weight):
        super().__init__()
        self.weight=weight
        self.gaussian1 = torch.distributions.Normal(0,sigma_1.to('cuda:0'))
        self.gaussian2 = torch.distributions.Normal(0,sigma_2.to('cuda:0'))
    
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.weight * prob1 + (1-self.weight) * prob2)).sum()

class BayesianLayer(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.weight_mu = nn.Parameter(torch.Tensor(self.channel_out, self.channel_in).uniform_(-0.2, 0.2).to('cuda:0'))
        self.weight_rho = nn.Parameter(torch.Tensor(self.channel_out, self.channel_in).uniform_(-10, -9).to('cuda:0'))
        self.bias_mu = nn.Parameter(torch.Tensor(self.channel_out).uniform_(-0.2, 0.2).to('cuda:0'))
        self.bias_rho = nn.Parameter(torch.Tensor(self.channel_out).uniform_(-10, -9).to('cuda:0'))
        self.weight_prior = MixGaussian(torch.FloatTensor([math.exp(0)]),torch.FloatTensor([math.exp(-4)]),0.5)
        self.bias_prior = MixGaussian(torch.FloatTensor([math.exp(0)]),torch.FloatTensor([math.exp(-6)]),0.5)
        self.log_prior = 0
        self.log_posterior = 0

    def forward(self, input,sample=True, calculate=True):
        if sample:
            weight_epsilon = torch.distributions.Normal(0,1).sample(self.weight_rho.size()).to('cuda:0')
            bias_epsilon = torch.distributions.Normal(0,1).sample(self.bias_rho.size()).to('cuda:0')
            weight = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * weight_epsilon 
            bias = self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        if calculate:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            weight_log_posterior=(-math.log(math.sqrt(2 * math.pi))- torch.log(torch.log1p(torch.exp(self.weight_rho)))- ((weight - self.weight_mu) ** 2) / (2 * torch.log1p(torch.exp(self.weight_rho)) ** 2)).sum()
            bias_log_posterior=(-math.log(math.sqrt(2 * math.pi))- torch.log(torch.log1p(torch.exp(self.bias_rho)))- ((bias - self.bias_mu) ** 2) / (2 * torch.log1p(torch.exp(self.bias_rho)) ** 2)).sum()
            self.log_posterior = weight_log_posterior + bias_log_posterior

        return F.linear(input, weight, bias)


class BNN(nn.Module):
    def __init__(self,channel_in):
        super().__init__()
        self.channel_in=channel_in
        self.l1 = BayesianLayer(self.channel_in, 256)
        self.l2 = BayesianLayer(256, 128)
        self.l3 = BayesianLayer(128, 4)
    
    def forward(self, x, sample=False,cal=True):
        x = x.view(-1, self.channel_in)
        x = F.relu(self.l1(x, sample,cal))
        x = F.relu(self.l2(x, sample,cal))
        x = F.relu(self.l3(x, sample,cal))
        return x
    
    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior
    
    def log_posterior(self):
        return self.l1.log_posterior + self.l2.log_posterior + self.l2.log_posterior
    
    def mse_l(self,x1,x2):
        return F.mse_loss(x1, x2, reduction='sum')
    