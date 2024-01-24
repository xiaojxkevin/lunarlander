import base64
from collections import deque

import imageio
import pandas as pd
from matplotlib import pyplot as plt

import wandb
import tensorflow as tf
from keras.layers import Input, Dense
from keras.backend import clear_session
from memory_profiler import profile
import gym
import argparse
import numpy as np

# tf.keras.backend.set_floatx('float64')

class Actor:
    def __init__(self, state_dim, action_dim, actor_lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])

    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(
            actions, logits, sample_weight=tf.stop_gradient(advantages))
        return policy_loss

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(
                actions, logits, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim, critic_lr):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Agent:
    def __init__(self, model, num_actions, input_dims, actor_lr=0.0005, critic_lr=0.001,
                 discount_factor=0.995, batch_size=4):

        self.action_dim = num_actions
        self.state_dim = input_dims
        self.actor = Actor(self.state_dim, self.action_dim, actor_lr)
        self.critic = Critic(self.state_dim, critic_lr)
        self.model = model
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.target_ave_scores = 500
        self.max_num_timesteps = 1000
        self.num_p_av = 100

    def td_target(self, reward, next_state, done):
        if done:
            return reward
        v_value = self.critic.model(
            np.reshape(next_state, [1, self.state_dim]))
        return np.reshape(reward + self.discount_factor * v_value[0], [1, 1])

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    # @profile
    def train(self, env, max_episodes=1000, graph=True):
        wandb.init(name='PPO', project="deep-rl-tf2")
        episodes_history, total_point_history, avg_point_history, target_history = [], [], [], []
        for ep in range(max_episodes):
            state_batch = []
            action_batch = []
            td_target_batch = []
            advatnage_batch = []
            episode_reward, done = 0, False

            state, _ = env.reset()
            for t in range(self.max_num_timesteps):
                # self.env.render()
                state = np.reshape(state, [1, self.state_dim])
                probs = self.actor.model(state)
                probs = np.array(probs)
                probs /= probs.sum()

                action = np.random.choice(self.action_dim, p=probs[0])

                next_state, reward, done, _, _ = env.step(action)

                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                td_target = self.td_target(reward * 0.01, next_state, done)
                advantage = self.advatnage(
                    td_target, self.critic.model(state))

                # actor_loss = self.actor.train(state, action, advantage)
                # critic_loss = self.critic.train(state, td_target)

                state_batch.append(state)
                action_batch.append(action)
                td_target_batch.append(td_target)
                advatnage_batch.append(advantage)

                if len(state_batch) >= self.batch_size or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    td_targets = self.list_to_batch(td_target_batch)
                    advantages = self.list_to_batch(advatnage_batch)

                    actor_loss = self.actor.train(states, actions, advantages)
                    critic_loss = self.critic.train(states, td_targets)

                    state_batch = []
                    action_batch = []
                    td_target_batch = []
                    advatnage_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]
                if done:
                    break

            episodes_history.append(ep)
            target_history.append(self.target_ave_scores)
            total_point_history.append(episode_reward)
            av_latest_points = np.mean(total_point_history[-self.num_p_av:])
            avg_point_history.append(av_latest_points)

            wandb.log({'target_ave_scores': 300, 'epoch': ep})
            wandb.log({'total_points': episode_reward, 'epoch': ep})
            wandb.log({'av_latest_points': av_latest_points, 'epoch': ep})

            print(
                f"\rEpisode {ep + 1} | Total point average of the last {self.num_p_av} episodes: {av_latest_points:.2f}",
                end="")

            if (ep + 1) % self.num_p_av == 0:
                print(
                    f"\rEpisode {ep + 1} | Total point average of the last {self.num_p_av} episodes: {av_latest_points:.2f}")

            # We will consider that the environment is solved if we get an
            # average of 200 points in the last 100 episodes.
            if av_latest_points >= self.target_ave_scores or ep + 1 == max_episodes:
                print(f"\n\nEnvironment solved in {ep + 1} episodes!")
                self.actor.model.save('saved_networks/' + self.model + '/lunar_lander_model_actor.h5')
                self.critic.model.save('saved_networks/' + self.model + '/lunar_lander_model_critic.h5')
                break

            if graph:
                df = pd.DataFrame({'x': episodes_history, 'Score': total_point_history,
                                   'Average Score': avg_point_history, 'Solved Requirement': target_history})

                plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
                plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
                         label='AverageScore')
                plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
                         label='Solved Requirement')
                plt.legend()
                plt.savefig('LunarLander_Train_' + self.model + '.png')

    def test_create_video(self, env, filename, model_name, fps=30):
        self.actor.model = tf.keras.models.load_model(model_name)
        with imageio.get_writer(filename, fps=fps) as video:
            done = False
            state, _ = env.reset()
            frame = env.render()
            video.append_data(frame)
            episode_score = 0
            for t in range(self.max_num_timesteps):
                state = np.reshape(state, [1, self.state_dim])
                probs = self.actor.model(state)
                action = np.argmax(probs.numpy()[0])
                state, reward, done, _, _ = env.step(action)
                episode_score += reward
                frame = env.render()
                video.append_data(frame)
                if done:
                    break
            print(f'episode_score:', episode_score)
            """Embeds an mp4 file in the notebook."""
            video = open(filename, 'rb').read()
            b64 = base64.b64encode(video)
            tag = '''
            <video width="840" height="480" controls>
            <source src="data:video/mp4;base64,{0}" type="video/mp4">
            Your browser does not support the video tag.
            </video>'''.format(b64.decode())
            # return IPython.display.HTML(tag)


def main():
    # myModel = 'A2C'
    # env = gym.make('LunarLander-v2', render_mode='rgb_array')
    # agent = Agent('A2C', 4, 8, actor_lr=0.0003, critic_lr=0.0005,
    #               discount_factor=0.995, batch_size=4)
    # myModel = 'PPO'
    # env = gym.make('LunarLander-v2', render_mode='rgb_array')
    # agent = Agent('PPO', 4, 8, actor_lr=0.0003, critic_lr=0.0005,
    #               discount_factor=0.995, batch_size=4)
    myModel = 'SAC'
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    agent = Agent('SAC', 4, 8, actor_lr=0.0003, critic_lr=0.0005,
                  discount_factor=0.995, batch_size=4)
    agent.train(env)
    agent.test_create_video(env, filename='Lunar_Lander_videos/lunar_lander' + myModel + '.mp4',
                            model_name='saved_networks/' + myModel + '/lunar_lander_model_actor.h5', fps=30)


if __name__ == "__main__":
    main()

