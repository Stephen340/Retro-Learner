# NOTE: This file is solely used to visualize an agent playing the game.  No training / testing logic
# is in this file.  See MarioToSonic.py for the full training logic.

import gc
import os
import random
import sys
import time
from collections import deque
from copy import deepcopy

import numpy as np
import retro
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import transforms as T
from torch.optim import AdamW
from gym import ObservationWrapper
from gym.spaces import Box
from PIL import Image
import matplotlib.pyplot as plt
from gym import Wrapper

MODEL_PATH = 'MarioFinetunedtoSonic.pth'

def convertAct(action):
    if action[1] or action[8]:  # There are three jump buttons; capture all in action[0]
        action[0] = 1

    if action[6]:
        return 4  # left

    if action[0]:  # Jump is pressed
        if action[7]:
            return 3  # Jump, Right
        elif action[5]:
            return 0  # Jump, Down
        else:
            return 2  # Jump

    if action[5]:  # Down is pressed
        if action[7]:
            return 5  # Down, Right
        else:
            return 0  # Down

    if action[7]:
        return 1  # right

    return 0  # no action / none of the above


action_switch = {
    0: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Down
    1: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Right
    2: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B
    3: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Right, B
    4: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Left
    5: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  # Right, Down
}

def convertActBack(actionID):
    return action_switch[actionID]

class FrameSkip(Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ResizeObservation(ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # Convert numpy array to PIL image
        observation = Image.fromarray(observation)

        # Apply transformations: resize to specified shape and grayscale, then normalize and convert to tensor
        transform = T.Compose([
            T.Resize(self.shape),
            T.Grayscale(),
            T.ToTensor(),  # Converts to [0, 1] tensor
            T.Lambda(lambda x: x * 255)  # Rescale back to [0, 255] range as a tensor
        ])
        observation = transform(observation)
        return observation


N_ACTIONS = 6


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, N_ACTIONS),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DQN:
    def __init__(self, env, movie):
        # Create CNN
        self.movie = movie
        self.env = env
        self.model = CustomCNN()  # Action space
        self.chunk_size = 4
        self.sequence_buffer = deque(maxlen=4)  # Temporary buffer to store chunks

        self.movie.step()
        self.env.initial_state = self.movie.get_state()

        # Create target Q-network
        self.target_model = deepcopy(self.model)

        # Set up the optimizer
        self.optimizer = AdamW(
            self.model.parameters(), lr=.001, amsgrad=True
        )
        # Define the loss function
        self.loss_fn = nn.SmoothL1Loss()

        # Freeze target network parameters
        for p in self.target_model.parameters():
            p.requires_grad = False

        # Replay buffer
        self.replay_memory = deque(maxlen=1000000)  # Max replay size

        # Number of training steps so far
        self.n_steps = 0
        self.epsilon = 0.9  # Start with full exploration
        self.epsilon_min = 0.05  # Minimum exploration
        self.epsilon_decay = 0.99999  # Decay rate

    def load_model(self, path):
        """
        Load the model's state_dict, optimizer state_dict, and target_model state_dict from a file.
        """
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        print(f"Model loaded from {path}")

    def epsilon_greedy(self, state):
        nA = N_ACTIONS
        # Assuming `state` is already in the right format (1, 84, 84)
        state = state.float().unsqueeze(0)  # Add batch dimension for model
        action_values = self.model(state)
        if random.random() < self.epsilon:
            # Random action
            chosen_action = np.random.choice(np.arange(nA))
        else:
            chosen_action = torch.argmax(action_values).item()
        return chosen_action

    def play_episode(self):
        state = self.env.reset()
        for _ in range(131072):  # Self.options.steps
            # If a movie is loaded, step through the movie instead
            chosen_action_id = self.epsilon_greedy(state)
            chosen_action = convertActBack(chosen_action_id)

            # step through the environment with the chosen action
            next_state, reward, done, info = self.env.step(chosen_action)

            self.env.render()
            time.sleep(0.01)

            # Update variables for next step
            state = next_state

            if done:
                break

    def __str__(self):
        return "DQN"



movie = retro.Movie('movies/sonic_movie.bk2')
env = retro.make(
    game=movie.get_game(),
    state=None,
    use_restricted_actions=retro.Actions.ALL,
    players=movie.players,
)
env.initial_state = movie.get_state()
env.reset()
env = FrameSkip(env, skip=4)  # Apply frame skipping
env = ResizeObservation(env, 84)  # Resize observation

# Initialize DQN only once
dqn = DQN(env, movie)

# Load a model
if not os.path.exists(MODEL_PATH):
    print("ISSUE: COULD NOT LOAD MODEL; IS THE FILE NAME CORRECT?")
    sys.exit()

dqn.load_model(MODEL_PATH)

for i in range(40000):
    print(f"Episode {i + 1}")
    env.initial_state = movie.get_state()
    dqn.env.reset()
    dqn.play_episode()
