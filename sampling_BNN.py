import torch
import torch.nn.functional as F
import gym
import os
from PIL import Image
import numpy as np

from model import QLearning

epochs = 10000
load_path="./results/ql/v2.pth"
MAX_ITERATION = 10000
render = False
required = 100
env = gym.make('LunarLander-v2',
                render_mode = "rgb_array")

policy = QLearning()
policy.load_state_dict(torch.load(load_path))

records = []
count = 0
with torch.no_grad():
    for epoch in range(1, epochs+1):
        obser, _ = env.reset()
        running_reward = 0
        record = []
        action=0
        for t in range(1, MAX_ITERATION+1):
            out = policy(obser)
            action_probs = F.softmax(out, dim=0)
            action_n = int(torch.argmax(action_probs, dim=0))
            r = list(obser)
            r.append(action)
            for i in range(4):
                if(action_n==i):
                    r.append(1)
                else:
                    r.append(0)
            record.append(r)
            obser, reward, terminated, truncated, _ = env.step(action_n)
            action=action_n
            running_reward += reward
            if render:
                img = env.render()
                img = Image.fromarray(img)
                gif_dir = "./gifs/test/{:02d}".format(epoch)
                if not os.path.exists(gif_dir):
                    os.makedirs(gif_dir)
                img.save(os.path.join(gif_dir, "{:04d}.png").format(t))

            if terminated or truncated:
                break

        if running_reward >= 200:
            count += 1
            for r in record:
                records.append(r)
        
        if count == required:
            break

        # recored_actions.append(actions)
        print('Episode {}\tReward: {}'.format(epoch, running_reward))

env.close()

np.save("./sample2.npy", np.asarray(records))
     