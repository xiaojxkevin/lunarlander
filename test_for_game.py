import torch
import torch.nn.functional as F
import numpy as np
import gym
from PIL import Image
import os
from lunarLander import Game

from model_for_game import ActorCritic, QLearning, OfflineClassifier

def create_gif(env, epoch:int, t:int):
    img = env.render()
    img = Image.fromarray(img)
    gif_dir = "./gifs/test/{:02d}".format(epoch)
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    img.save(os.path.join(gif_dir, "{:04d}.png").format(t))


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
            # create_gif(env, epoch, t)

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
            # create_gif(env, epoch, t)

            if terminated or truncated:
                break

        # print('Episode {}\tReward: {}'.format(epoch, running_reward))
        score_list.append(running_reward)
        
    print(np.mean(score_list), np.std(score_list), np.max(score_list), np.min(score_list))
    env.close()

@torch.no_grad()
def test_ql_game(epochs=10, load_path="./ckpts/game.pth"):
    # Set up Configs
    MAX_ITERATION = 10000

    policy = QLearning()
    policy.load_state_dict(torch.load(load_path))

    score_list = []

    for epoch in range(1, epochs+1):

        game = Game()
        game.resetGame()
        state = game.get_ship_info()
        state = np.array(state, dtype=np.float32)

        running_reward = 0
        for t in range(1, MAX_ITERATION+1):
            out = policy(state)
            action_probs = F.softmax(out, dim=0)
            action = torch.argmax(action_probs, dim=0)
            next_state, reward, end = game.execute_action(action)
            next_state = np.array(next_state, dtype=np.float32)
            running_reward += reward
            # create_gif(env, epoch, t)

            if end:
                break

        # print('Episode {}\tReward: {}'.format(epoch, running_reward))
        score_list.append(running_reward)
        
    print(np.mean(score_list), np.std(score_list), np.max(score_list), np.min(score_list))

@torch.no_grad()
def test_offline(epochs=100, load_path="./ckpts/v3.pth"):
    # Set up Configs
    MAX_ITERATION = 10000
    env = gym.make('LunarLander-v2',
                   render_mode = "rgb_array")

    policy = OfflineClassifier()
    policy.load_state_dict(torch.load(load_path, map_location="cpu"))

    score_list = []

    for epoch in range(1, epochs+1):
        obser, _ = env.reset()
        obser = torch.from_numpy(obser).unsqueeze(0)
        # assert False
        running_reward = 0
        for t in range(1, MAX_ITERATION+1):
            out = policy(obser)
            action_probs = F.softmax(out, dim=1)
            action = torch.argmax(action_probs, dim=1)
            obser, reward, terminated, truncated, _ = env.step(int(action))
            obser = torch.from_numpy(obser).unsqueeze(0)            
            running_reward += reward
            if epoch % 20 == 0:
                create_gif(env, epoch, t)

            if terminated or truncated:
                break

        # print('Episode {}\tReward: {}'.format(epoch, running_reward))
        score_list.append(running_reward)
        
    print(np.mean(score_list), np.std(score_list), np.max(score_list), np.min(score_list))
    env.close()

if __name__ == '__main__':
    #test_offline()
    # test_ac()
    test_ql_game()
