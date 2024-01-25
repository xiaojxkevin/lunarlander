import torch
import torch.nn.functional as F
import gym
import os
from PIL import Image
import numpy as np

from model import QLearning

epochs = 10000
load_path="./ckpts/v2.pth"
MAX_ITERATION = 10000
render = True
required = 100
env = gym.make('LunarLander-v2',
                render_mode = "rgb_array")

policy = ActorCritic()
# policy.load_state_dict(torch.load(load_path))

records = []
count = 0
for epoch in range(1, epochs+1):
    obser, _ = env.reset()
    running_reward = 0
    record = []
    for t in range(1, MAX_ITERATION+1):
        action = policy(obser)
        r = list(obser)
        r.append(action)
        record.append(r)
        obser, reward, terminated, truncated, _ = env.step(action)
        running_reward += reward
        if render and epoch%100==0:
            img = env.render()
            img = Image.fromarray(img)
            gif_dir = "./gifs/pretrained/{:02d}".format(epoch)
            if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)
            img.save(os.path.join(gif_dir, "{:04d}.png".format(t)))
            print (obser,action,reward)
        if terminated or truncated:
            break

        # recored_actions.append(actions)
    print('Episode {}\tReward: {}'.format(epoch, running_reward))

env.close()

np.save("./actions/v2_{:05d}".format(required), np.asarray(records))
     