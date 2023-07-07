from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from a2c import A2C

import gymnasium as gym

# environment hyperparams
randomize_domain = False
actor_lr = 0.001
critic_lr = 0.005
n_showcase_episodes = 3

load_weights = True

actor_weights_path = "weights/actor_weights.h5"
critic_weights_path = "weights/critic_weights.h5"

def main():
     # set the device
    use_cuda = True
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    for episode in range(n_showcase_episodes):
        print(f"starting episode {episode}...")

        # create a new sample environment to get new random parameters
        if randomize_domain:
            env = gym.make(
                "LunarLander-v2",
                render_mode="human",
                gravity=np.clip(
                    np.random.normal(loc=-10.0, scale=2.0), a_min=-11.99, a_max=-0.01
                ),
                enable_wind=np.random.choice([True, False]),
                wind_power=np.clip(
                    np.random.normal(loc=15.0, scale=2.0), a_min=0.01, a_max=19.99
                ),
                turbulence_power=np.clip(
                    np.random.normal(loc=1.5, scale=1.0), a_min=0.01, a_max=1.99
                ),
                max_episode_steps=500,
            )
        else:
            env = gym.make("LunarLander-v2", render_mode="human", max_episode_steps=500)


        obs_shape = env.observation_space.shape[0]
        action_shape = env.action_space.n

        if load_weights:
            agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, 1)

        agent.actor.load_state_dict(torch.load(actor_weights_path))
        agent.critic.load_state_dict(torch.load(critic_weights_path))
        agent.actor.eval()
        agent.critic.eval()

        # get an initial state
        state, info = env.reset()

        total_reward = 0

        # play one episode
        done = False
        while not done:
            # select an action A_{t} using S_{t} as input for the agent
            with torch.no_grad():
                action, _, _, _ = agent.select_action(state[None, :])

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            state, reward, terminated, truncated, info = env.step(action.item())

            total_reward += reward

            # update if the environment is done
            done = terminated or truncated
        
        print("Total reward:", total_reward)

        env.close()


if __name__ == '__main__':
    main()
