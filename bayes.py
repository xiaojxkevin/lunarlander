import torch
import gym
import os
from PIL import Image
import json
import numpy as np
import random
import argparse
from model import ActorCritic

epochs = 100000
MAX_ITERATION = 10000
render = True
required = 100
opts = argparse.ArgumentParser()
opts.add_argument("--random_seed", type=int, default=3407)
opts.add_argument("--id", type=str, default='LunarLander-v2')

args = opts.parse_args()
print(args)

torch.manual_seed(args.random_seed)

env = gym.make(args.id,
               render_mode = "rgb_array")
env.action_space.seed(args.random_seed)
lr=0.9
action_prob=np.zeros((3,4,4))
action_prob[:,:,:]=0.25
for epoch in range(1, epochs+1):
    obser,_=env.reset()
    running_reward = 0
    record = []
    if epoch%1000==0:
        lr=lr*0.95
    action=0
    temp_prob=action_prob.copy()
    for t in range(1, MAX_ITERATION+1):
        r=list(obser)
        rand=random.random()
        count=0
        for action_n in range(4):
            if(r[1]<0.5):
                if rand>=count+action_prob[2][action][action_n]:
                    count+=action_prob[2][action][action_n]
                else:
                    break
            elif(r[1]<1 and r[1]>=0.5):
                if rand>=count+action_prob[1][action][action_n]:
                    count+=action_prob[1][action][action_n]
                else:
                    break
            else:
                if rand>=count+action_prob[0][action][action_n]:
                    count+=action_prob[0][action][action_n]
                else:
                    break
        obser, reward, terminated, truncated, _ = env.step(action_n)
        if(r[1]<0.5):
            action_prob[2][action][action_n]+=0.002*lr*reward
            if(action_n==3):
                action_prob[2][action][1]-=0.002*lr*reward       
            elif(action_n==1):
                action_prob[2][action][3]-=0.002*lr*reward
            elif(action_n==2):
                action_prob[2][action][0]-=0.002*lr*reward
            else:
                action_prob[2][action][2]-=0.002*lr*reward
        elif(r[1]<1 and r[1]>=0.5):
            action_prob[1][action][action_n]+=0.04*lr*reward
            if(action_n==3):
                action_prob[1][action][1]-=0.04*lr*reward       
            elif(action_n==1):
                action_prob[1][action][3]-=0.04*lr*reward
            elif(action_n==2):
                action_prob[1][action][0]-=0.04*lr*reward
            else:
                action_prob[1][action][2]-=0.04*lr*reward
        else:
            action_prob[0][action][action_n]+=0.08*lr*reward
            if(action_n==3):
                action_prob[0][action][1]-=0.08*lr*reward       
            elif(action_n==1):
                action_prob[0][action][3]-=0.08*lr*reward
            elif(action_n==2):
                action_prob[0][action][0]-=0.08*lr*reward
            else:
                action_prob[0][action][2]-=0.08*lr*reward
        action=action_n 
        running_reward += reward
        if render and epoch%500==0:
            img = env.render()
            img = Image.fromarray(img)
            gif_dir = "./gifs/bayes/{:02d}".format(epoch)
            if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)
            img.save(os.path.join(gif_dir, "{:04d}.png".format(t)))

        if terminated or truncated:
            break

    # if count == required:
    #     break

        # recored_actions.append(actions)
    
    if epoch%100==0:
        print('Episode {}\tReward: {}'.format(epoch, running_reward))
        print(action_prob)
    for i in range(3):
        for j in range(4):
            action_prob[i,j,:]=(0.95*temp_prob[i,j,:]+0.05*action_prob[i,j,:])/(0.95*np.sum(temp_prob[i,j,:])+0.05*np.sum(action_prob[i,j,:]))

    # 
    # print(action_prob)
    # if epoch%500==0:
    #     print(action_prob)
env.close()