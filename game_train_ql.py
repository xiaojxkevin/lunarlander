# Description: This file is used to train the agent using the Q-Learning algorithm.
import os
import torch
import time
import numpy as np
import torch.optim as optim
import argparse
import gym
from PIL import Image
from tensorboardX import SummaryWriter
from ship import Ship
from terrain import Terrain
from lunarLander import *

from model import QLearning

opts = argparse.ArgumentParser()
opts.add_argument("--random_seed", type=int, default=3407)
opts.add_argument("--id", type=str, default='LunarLander-v2')
opts.add_argument("--epochs", type=int, default=int(1e4))
opts.add_argument("--batch_size", type=int, default=64)
opts.add_argument("--lr", type=float, default=1e-3)
opts.add_argument("--gamma", type=float, default=0.96)
opts.add_argument("--epsilon", type=float, default=1)
opts.add_argument("--epsilon_min", type=float, default=0.01)
opts.add_argument("--epsilon_decay", type=float, default=0.97)
opts.add_argument("--buffer_size", type=int, default=10000)
opts.add_argument("--max_iteration", type=int, default=int(1e4))
args = opts.parse_args()
print(args)

torch.manual_seed(args.random_seed)

policy = QLearning(maxlen=args.buffer_size, 
                   batch_size=args.batch_size, 
                   gamma=args.gamma, 
                   epsilon=args.epsilon)

optimizer = optim.Adam(policy.parameters(), lr=args.lr)

current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
writer = SummaryWriter(log_dir=os.path.join("./log", "1.26q_game2"))

for epoch in range(1, args.epochs+1):
    reward_for_episode = 0
    
    game = Game()
    game.resetGame()
    state = game.get_ship_info()
    state = np.reshape(state, [1, 7])

    for t in range(1, args.max_iteration+1):
        received_action = policy.choose_action(state)
        # assert False, "{} {}".format(type(received_action), received_action)
        next_state, reward, end = game.execute_action(received_action)
        # assert False, "{} {}".format(next_state.shape, reward)
        next_state = np.reshape(next_state, [1, 7])
        # Store the experience in memory
        done = float(end)
        policy.add_to_memory(state, received_action, reward, next_state, done)
        # add up rewards
        reward_for_episode += reward
        state = next_state
        loss = policy.compute_loss()
        if loss:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if end:
            break

        if epoch % 20 == 0:
            gif_dir = "./gifs/game/train2/{:02d}".format(epoch)
            if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)
            save_path = os.path.join(gif_dir, "{:04d}.png".format(t))
            game.customDraw(save_path)


    writer.add_scalar("Score For Training with Q-Learning", reward_for_episode, epoch)
    print(epoch, reward_for_episode)
    policy.rewards_list.append(reward_for_episode)

    # Decay the epsilon after each experience completion
    if policy.epsilon > args.epsilon_min:
        policy.epsilon *= args.epsilon_decay

    # Check for breaking condition
    last_rewards_mean = np.mean(policy.rewards_list[-20:])
    if last_rewards_mean > 10000000 or epoch in [50, 100, 200, 500, 1000, 2000, 2500, 3000, 3500, 4000]:
        save_path = "./ckpts/game_final.pth"
        torch.save(policy.state_dict(), save_path)
        print("############## Finish Training ##############")
        if epoch == 4000:
            break
