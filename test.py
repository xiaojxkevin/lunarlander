import torch
import torch.nn.functional as F
import numpy as np
import gym
from PIL import Image
import os
import matplotlib.pyplot as plt
from model import ActorCritic, QLearning, OfflineClassifier, BNN, Bayes

def create_gif(env, epoch:int, t:int):
    img = env.render()
    img = Image.fromarray(img)
    gif_dir = "./gifs/test_bnn/{:02d}".format(epoch)
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    img.save(os.path.join(gif_dir, "{:04d}.png".format(t)))


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


@torch.no_grad()
def test_BNN(epochs=100, load_path="./ckpt/v7.pth"):
    # Set up Configs
    MAX_ITERATION = 10000
    env = gym.make('LunarLander-v2',
                   render_mode = "rgb_array")

    policy = BNN(9).to('cuda:0')
    policy.load_state_dict(torch.load(load_path, map_location="cpu"))

    score_list = []
    epoch_l=[]
    for epoch in range(1, epochs+1):
        obser, _ = env.reset()
        obser=np.append(obser,values=0.0)
        obser = torch.from_numpy(obser).unsqueeze(0).to('cuda:0').to(torch.float32)
        # assert False
        running_reward = 0
        for t in range(1, MAX_ITERATION+1):
            out = policy(obser,sample=True,cal=False)
            action_probs = F.softmax(out, dim=1)
            action = torch.argmax(action_probs, dim=1)
            obser, reward, terminated, truncated, _ = env.step(int(action))
            obser=np.append(obser,values=int(action.item()))
            obser = torch.from_numpy(obser).unsqueeze(0).to('cuda:0').to(torch.float32)            
            running_reward += reward
            # if epoch % 20 == 0:
            #     create_gif(env, epoch, t)

            if terminated or truncated:
                break
        if epoch % 20 == 0:
            print('Episode {}\tReward: {}'.format(epoch, running_reward))
        # print('Episode {}\tReward: {}'.format(epoch, running_reward))
        score_list.append(running_reward)
        epoch_l.append(epoch)
    plt.figure(1)
    plt.plot(epoch_l,score_list,label='Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score_BNN')
    plt.savefig('./results/Score_BNN.png')
    print(np.mean(score_list), np.std(score_list), np.max(score_list), np.min(score_list))
    env.close()

@torch.no_grad()
def test_bayes(load_path="./ckpt/bayes_output.npy"):
    # Set up Configs
    MAX_ITERATION = 10000
    env = gym.make('LunarLander-v2',
                   render_mode = "rgb_array")
    obser, _ = env.reset()
    action=0
    prob=np.load(load_path)
    for t in range(1, MAX_ITERATION+1):
        if(obser[1]<0.5):
            action_n=np.argmax(prob[2,action,:])
        elif(obser[1]>=0.5 and obser[1]<1):
            action_n=np.argmax(prob[1,action,:])
        else:
            action_n=np.argmax(prob[0,action,:])
        _, _, terminated, truncated, _ = env.step(int(action))
        img = env.render()
        img = Image.fromarray(img)
        gif_dir = "./gifs/test_bayes"
        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
        img.save(os.path.join(gif_dir, "{:04d}.png").format(t))
        action=action_n
        if terminated or truncated:
            break

        # print('Episode {}\tReward: {}'.format(epoch, running_reward))
    env.close()


@torch.no_grad()
def test_bayes_net(load_path="./ckpt/LunarLander_bayes.pth"):
    # Set up Configs
    epochs=10
    MAX_ITERATION = 10000
    env = gym.make('LunarLander-v2',
                   render_mode = "rgb_array")
    obser, _ = env.reset()
    running_reward = 0
    policy=Bayes()
    policy.load_state_dict(torch.load(load_path))
    for epoch in range(1, epochs+1):
        obser, _ = env.reset()
        running_reward = 0
        action=0
        for t in range(1, MAX_ITERATION+1):
            obser=np.append(obser,values=action)
            action,_ = policy(obser)
            obser, reward, terminated, truncated, _ = env.step(action)
            running_reward += reward
            img = env.render()
            img = Image.fromarray(img)
            gif_dir = "./gifs/test_bayes_net/{:02d}".format(epoch)
            if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)
            img.save(os.path.join(gif_dir, "{:04d}.png".format(t)))

            if terminated or truncated:
                break

        # print('Episode {}\tReward: {}'.format(epoch, running_reward))
    env.close()
if __name__ == '__main__':
    test_BNN()
    # test_ac()
    # test_ql()

