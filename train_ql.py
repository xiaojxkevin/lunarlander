import os
import torch
import time
import numpy as np
import torch.optim as optim
import argparse
import gym
from PIL import Image
from tensorboardX import SummaryWriter

from model import QLearning

opts = argparse.ArgumentParser()
opts.add_argument("--random_seed", type=int, default=3407)
opts.add_argument("--id", type=str, default='LunarLander-v2')
opts.add_argument("--epochs", type=int, default=int(1e4))
opts.add_argument("--batch_size", type=int, default=64)
opts.add_argument("--lr", type=float, default=1e-3)
opts.add_argument("--gamma", type=float, default=0.99)
opts.add_argument("--epsilon", type=float, default=1)
opts.add_argument("--epsilon_min", type=float, default=0.01)
opts.add_argument("--epsilon_decay", type=float, default=0.95)
opts.add_argument("--buffer_size", type=int, default=10000)
opts.add_argument("--max_iteration", type=int, default=int(1e4))
args = opts.parse_args()
print(args)

torch.manual_seed(args.random_seed)

env = gym.make(args.id,
               render_mode = "rgb_array")
env.action_space.seed(args.random_seed)

policy = QLearning(maxlen=args.buffer_size, 
                   batch_size=args.batch_size, 
                   gamma=args.gamma, 
                   epsilon=args.epsilon)

optimizer = optim.Adam(policy.parameters(), lr=args.lr)

current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
writer = SummaryWriter(log_dir=os.path.join("./log/q", current_time))

for epoch in range(1, args.epochs+1):
    reward_for_episode = 0
    state, _ = env.reset()
    state = np.reshape(state, [1, 8])
    for t in range(1, args.max_iteration+1):
        received_action = policy.choose_action(state)
        # assert False, "{} {}".format(type(received_action), received_action)
        next_state, reward, terminated, truncated, _ = env.step(received_action)
        # assert False, "{} {}".format(next_state.shape, reward)
        next_state = np.reshape(next_state, [1, 8])
        done = float(terminated or truncated)
        # Store the experience in memory
        policy.add_to_memory(state, received_action, reward, next_state, done)
        # add up rewards
        reward_for_episode += reward
        state = next_state
        loss = policy.compute_loss()
        if loss:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if terminated or truncated:
            break

        if epoch % 20 == 0:
            img = env.render()
            img = Image.fromarray(img)
            gif_dir = "./gifs/train/{:02d}".format(epoch)
            if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)
            img.save(os.path.join(gif_dir, "{:04d}.png").format(t))

    writer.add_scalar("Score For Training with Q-Learning", reward_for_episode, epoch)
    print(epoch, reward_for_episode)
    policy.rewards_list.append(reward_for_episode)

    # Decay the epsilon after each experience completion
    if policy.epsilon > args.epsilon_min:
        policy.epsilon *= args.epsilon_decay

    # Check for breaking condition
    last_rewards_mean = np.mean(policy.rewards_list[-20:])
    if last_rewards_mean > 200:
        save_path = "./ckpts/v2.pth"
        torch.save(policy.state_dict(), save_path)
        print("############## Finish Training ##############")
        break

env.close()