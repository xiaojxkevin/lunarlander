import torch
import torch.nn.functional as F
import numpy as np
import gym
from PIL import Image
import os

from model import ActorCritic, QLearning

@torch.no_grad()
def test_ac(epochs=100, load_path="./ckpts/v1.pth"):
    # Set up Configs
    MAX_ITERATION = 10000
    env = gym.make('LunarLander-v2',
                   render_mode = "rgb_array")

    policy = ActorCritic()
    policy.load_state_dict(torch.load(load_path))

    score_list = []

    for epoch in range(1, epochs+1):
        obser, _ = env.reset()
        running_reward = 0
        for t in range(1, MAX_ITERATION+1):
            action = policy(obser)
            obser, reward, terminated, truncated, _ = env.step(action)
            running_reward += reward
            # img = env.render()
            # img = Image.fromarray(img)
            # gif_dir = "./gifs/test/{:02d}".format(epoch)
            # if not os.path.exists(gif_dir):
            #     os.makedirs(gif_dir)
            # img.save(os.path.join(gif_dir, "{:04d}.png").format(t))

            if terminated or truncated:
                break

        # print('Episode {}\tReward: {}'.format(epoch, running_reward))
            score_list.append(running_reward)
        
    print(np.mean(score_list), np.std(score_list), np.max(score_list), np.min(score_list))
    env.close()

@torch.no_grad()
def test_ql(epochs=100, load_path="./ckpts/v2.pth"):
    # Set up Configs
    MAX_ITERATION = 10000
    env = gym.make('LunarLander-v2',
                   render_mode = "rgb_array")

    policy = QLearning()
    policy.load_state_dict(torch.load(load_path))

    score_list = []

    for epoch in range(1, epochs+1):
        obser, _ = env.reset()
        running_reward = 0
        for t in range(1, MAX_ITERATION+1):
            out = policy(obser)
            action_probs = F.softmax(out, dim=0)
            action = torch.argmax(action_probs, dim=0)
            obser, reward, terminated, truncated, _ = env.step(int(action))
            running_reward += reward
            # img = env.render()
            # img = Image.fromarray(img)
            # gif_dir = "./gifs/test/{:02d}".format(epoch)
            # if not os.path.exists(gif_dir):
            #     os.makedirs(gif_dir)
            # img.save(os.path.join(gif_dir, "{:04d}.png").format(t))

            if terminated or truncated:
                break

        # print('Episode {}\tReward: {}'.format(epoch, running_reward))
        score_list.append(running_reward)
        
    print(np.mean(score_list), np.std(score_list), np.max(score_list), np.min(score_list))
    env.close()

if __name__ == '__main__':
    test_ac()