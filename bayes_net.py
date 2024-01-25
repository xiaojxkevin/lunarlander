import torch
import gym
import os
from PIL import Image
import json
import numpy as np
import random
from model import Bayes
import torch.optim as optim
from test import test
import argparse

epochs = 120000
MAX_ITERATION = 10000
render = True
required = 100

opts = argparse.ArgumentParser()
opts.add_argument("--random_seed", type=int, default=3407)
opts.add_argument("--id", type=str, default='LunarLander-v2')
opts.add_argument("--epochs", type=int, default=int(1e4))
opts.add_argument("--lr", type=float, default=2e-2)
opts.add_argument("--betas", type=tuple, default=(0.9, 0.99))
opts.add_argument("--gamma", type=float, default=0.99)
opts.add_argument("--exit_score", type=int, default=4000)
opts.add_argument("--max_iteration", type=int, default=int(1e4))
args = opts.parse_args()
print(args)

torch.manual_seed(args.random_seed)

env = gym.make(args.id,
               render_mode = "rgb_array")
env.action_space.seed(args.random_seed)

policy=Bayes()
optimizer = optim.Adam(policy.parameters(), lr=5e-1, betas=(0.9, 0.99))
for epoch in range(1, epochs+1):
    obser,_=env.reset()
    running_reward = 0
    loss=0
    action_n=0
    for t in range(1, MAX_ITERATION+1):
        obser=np.append(obser,values=action_n)
        action_n = policy(obser)

        # for action in range(4):
        obser, reward, terminated, truncated, _ = env.step(action_n)
        running_reward += reward
        policy.reward_li.append(reward)
        # running_reward += reward
        if render and epoch%1000==0:
            img = env.render()
            img = Image.fromarray(img)
            gif_dir = "./gifs/bayes_net/{:02d}".format(epoch)
            if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)
            img.save(os.path.join(gif_dir, "{:04d}.png".format(t)))
        if terminated or truncated:
            break

    # if count == required:
    #     break
    optimizer.zero_grad()
    loss=policy.calculateLoss(0.9)
    loss.backward()
    optimizer.step()
    policy.clearMemory()
        # recored_actions.append(actions)
    if epoch%100==0:
        print('Episode {}\tReward: {}\tLoss: {}'.format(epoch, running_reward,loss.item()))
if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')
save_path = "./ckpt/LunarLander_bayes.pth"
torch.save(policy.state_dict(), save_path)
print("############## Finish Training ##############")
test(load_path=save_path)

    # action_prob[0,:]=0.8*temp_prob[0,:]+0.5*action_prob[0,:]/np.sum(action_prob[0,:])
    # action_prob[1,:]=0.8*temp_prob[1,:]+0.5*action_prob[1,:]/np.sum(action_prob[1,:])
    # action_prob[2,:]=0.8*temp_prob[2,:]+0.5*action_prob[2,:]/np.sum(action_prob[2,:])
    # if epoch%500==0:
    #     print(action_prob)
env.close()