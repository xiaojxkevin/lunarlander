import os
import torch
import time
import numpy as np
import torch.optim as optim
import argparse
import gym
from PIL import Image
from tensorboardX import SummaryWriter

from model import ActorCritic

opts = argparse.ArgumentParser()
opts.add_argument("--random_seed", type=int, default=3407)
opts.add_argument("--id", type=str, default='LunarLander-v2')
opts.add_argument("--epochs", type=int, default=int(1e4))
opts.add_argument("--lr", type=float, default=2e-2)
opts.add_argument("--betas", type=tuple, default=(0.9, 0.99))
opts.add_argument("--gamma", type=float, default=0.99)
opts.add_argument("--exit_score", type=int, default=200)
opts.add_argument("--max_iteration", type=int, default=int(1e4))
opts.add_argument("--w", type=float, default=1)
args = opts.parse_args()
print(args)

torch.manual_seed(args.random_seed)

env = gym.make(args.id,
               render_mode = "rgb_array")
env.action_space.seed(args.random_seed)

policy = ActorCritic()
optimizer = optim.Adam(policy.parameters(), lr=args.lr, betas=args.betas)

current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
writer = SummaryWriter(log_dir=os.path.join("./log/ac", current_time))

running_reward = []
for epoch in range(1, args.epochs+1):
    obser, _ = env.reset()

    score = 0
    for t in range(1, args.max_iteration+1):
        action = policy(obser)
        obser, reward, terminated, truncated, _ = env.step(action)

        policy.rewards.append(reward)
        score += reward

        if epoch % 200 == 0:
            img = env.render()
            img = Image.fromarray(img)
            gif_dir = "./gifs/train/{:02d}".format(epoch)
            if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)
            img.save(os.path.join(gif_dir, "{:04d}.png").format(t))

        if terminated or truncated:
            break

    running_reward.append(score)
    writer.add_scalar("Training score", score, epoch)
    # print("Epoch: {}, score: {}".format(epoch, score))
    optimizer.zero_grad()
    if epoch > 100 and epoch < 400 and epoch % 50 == 0:
        args.w *= 0.5
    elif epoch > 400:
        args.w = 0
    loss = policy.compute_loss(args.gamma, args.w)
    loss.backward()
    optimizer.step()
    policy.clearMemory()

    if np.mean(running_reward[-20:]) > args.exit_score:
        save_path = "./ckpts/v1.pth"
        torch.save(policy.state_dict(), save_path)
        print("############## Finish Training ##############")
        break

env.close()