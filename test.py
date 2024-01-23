import torch
import gym
from PIL import Image
import os

from model import ActorCritic

@torch.no_grad()
def test(epochs=5, load_path="./ckpts/LunarLander_0.02_0.9_0.99.pth"):
    # Set up Configs
    MAX_ITERATION = 10000
    env = gym.make('LunarLander-v2',
                   render_mode = "rgb_array")

    policy = ActorCritic()
    policy.load_state_dict(torch.load(load_path))

    for epoch in range(1, epochs+1):
        obser, _ = env.reset()
        running_reward = 0
        for t in range(1, MAX_ITERATION+1):
            action = policy(obser)
            obser, reward, terminated, truncated, _ = env.step(action)
            running_reward += reward
            img = env.render()
            img = Image.fromarray(img)
            gif_dir = "./gifs/test/{:02d}".format(epoch)
            if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)
            img.save(os.path.join(gif_dir, "{:04d}.png").format(t))

            if terminated or truncated:
                break

        print('Episode {}\tReward: {}'.format(epoch, running_reward))
    env.close()

if __name__ == '__main__':
    test()