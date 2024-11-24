import gc
import os
import random
from collections import deque
from copy import deepcopy

import numpy as np
import retro
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from sympy.physics.units import action
from torchvision import transforms
from torch.optim import AdamW
from PIL import Image
from torchvision import transforms as T
from gym import ObservationWrapper
from gym.spaces import Box
from gym import Wrapper


total_reward = []


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
        # Ensure observation is a numpy array
        if isinstance(observation, torch.Tensor):
            observation = observation.cpu().numpy()
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


def convertAct(action):
    if action[-1] == 1:  # Jumping
        if action[-2] == 1:  # Moving right
            return 3
        elif action[-3] == 1:  # Moving left
            return 5
        else:
            return 2  # Jumping
    else:
        if action[-2] == 1:  # Not jumping moving right
            return 1
        elif action[-3] == 1:  # Not jumping moving left
            return 4
        else:
            return 0  # Standing still


def convertActBack(actionID):
    if actionID == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif actionID == 1:
        return [0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif actionID == 2:
        return [0, 0, 0, 0, 0, 0, 0, 0, 1]
    elif actionID == 3:
        return [0, 0, 0, 0, 0, 0, 0, 1, 1]
    elif actionID == 4:
        return [0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif actionID == 5:
        return [0, 0, 0, 0, 0, 0, 1, 0, 1]
    else:
        # print("Unknown actionID", actionID)
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]


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

        if os.path.exists('mario_downsized2.pth'):
            self.load_model('mario_downsized2.pth')
        self.model.to('cuda')
        self.target_model.to('cuda')

        # Replay buffer
        self.replay_memory = deque(maxlen=1000000)  # Max replay size

        # Number of training steps so far
        self.n_steps = 0
        self.epsilon = 0.1  # Start with full exploration
        self.epsilon_min = 0.05  # Minimum exploration
        self.epsilon_decay = 0.99999975  # Decay rate
        self.leftward_counter = 0  # Each frame with left movement

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        """
        Save the model's state_dict, optimizer state_dict, and target_model state_dict to a file.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'target_model_state_dict': self.target_model.state_dict()
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load the model's state_dict, optimizer state_dict, and target_model state_dict from a file.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        print(f"Model loaded from {path}")

    def greedy(self, state):
        state = state.to('cuda').float().unsqueeze(0)  # Prepare the input
        action_values = self.model(state)  # Forward pass through the model

        # Extract the 1st and 3rd index values and compute their maximum
        indices = [1, 3]
        selected_values = action_values[0, indices]  # Assuming action_values is 2D [batch_size, num_actions]
        max_index = indices[torch.argmax(selected_values).item()]  # Get the index of the maximum value

        return max_index

    def epsilon_greedy(self, state):
        nA = N_ACTIONS
        # Assuming `state` is already in the right format (1, 84, 84)
        state = state.to('cuda').float().unsqueeze(0)  # Add batch dimension for model
        action_values = self.model(state)
        if random.random() < self.epsilon:
            # Random action
            chosen_action = np.random.choice(np.arange(nA))
            if chosen_action <= 2:
                chosen_action = 1
            else:
                chosen_action = 3
        else:
            chosen_action = torch.argmax(action_values).item()
        return chosen_action

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            # print(self.epsilon * self.epsilon_decay)
            self.epsilon *= self.epsilon_decay

    def compute_target_values(self, next_states, rewards, dones):
        next_q_vals = self.target_model(next_states)  # Shape should be (64 * 4, num_actions)
        best_next_q_vals = torch.max(next_q_vals, dim=1)[0]  # Shape should be (64 * 4)
        best_next_q_vals = best_next_q_vals.view(16, 4)  # Shape (64, 4)
        target = rewards + 0.9 * best_next_q_vals * (1 - dones)  # 0.9 is gamma

        return target

    def myreward(self, info, previnfo):
        # Calculate positions
        pxpos = previnfo['x_position2'] + previnfo['xscrollLo'] + 256 * previnfo['xscrollHi']
        xpos = info['x_position2'] + info['xscrollLo'] + 256 * info['xscrollHi']

        # Calculate changes
        dpos = xpos - pxpos  # Change in x position
        dtime = info['time'] - previnfo['time']  # Change in time

        # Check status flags
        isDead = info['player_state'] == 6 or info['player_state'] == 11
        if isDead:
            return -100
        isFlag = info['player_state'] == 4
        if isFlag:
            print('FFFFFFFFFFFFFFFLLLLLLLLLLLLLLLLLLLLLAAAAAAAAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGGGGGGGGGGGGGG')
            return 5000 + dpos * 2 if dpos > 0 else -5  # Reward for reaching the flag

        # Reward components
        rightward_reward = dpos * 2 if dpos > 0 else -5  # 2x reward for moving right, -5 for left
        time_bonus = 0.05 * dtime  # Smaller time reward

        # Final reward calculation
        reward = rightward_reward + time_bonus
        return reward

    def replay(self):
        if len(self.replay_memory) > 16:  # 64 is self.options.batch_size
            minibatch = random.sample(self.replay_memory, 16)
            states, actions, rewards, next_states, dones = [], [], [], [], []

            for chunk in minibatch:
                # Accumulate frames in the chunk for each transition element
                chunk_states, chunk_actions, chunk_rewards, chunk_next_states, chunk_dones = zip(*chunk)
                states.append(np.array(chunk_states))
                actions.append(np.array(chunk_actions))
                rewards.append(np.array(chunk_rewards))
                next_states.append(np.array(chunk_next_states))
                dones.append(np.array(chunk_dones))

            # Current Q-values
            states = np.array(states)  # Shape should be (batch_size, chunk_size, H, W, C)

            # Transpose to (batch_size, chunk_size, C, H, W) if necessary
            if states.shape[-1] == 1:  # Checking if the last dimension is the channel (C)
                states = states.transpose((0, 1, 4, 2, 3))

            # Convert numpy arrays to torch tensors
            states = torch.as_tensor(np.array(states), dtype=torch.float32).to('cuda')
            actions = torch.as_tensor(np.array(actions), dtype=torch.long).to('cuda')  # Using long for indexing
            rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32).to('cuda')
            next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32).to('cuda')
            dones = torch.as_tensor(np.array(dones), dtype=torch.float32).to('cuda')

            # Reshape to flatten batch and chunk for model input
            states = states.view(-1, 1, 84, 84)
            next_states = next_states.view(-1, 1, 84, 84)

            # Calculate current Q-values
            current_q = self.model(states)
            current_q = current_q.view(16, 4, -1)  # Reshape back to (batch_size, chunk_size, num_actions)

            # Gather Q-values for actions taken in replay memory
            actions = actions.unsqueeze(-1)  # Shape (64, 4, 1) to match current_q for gather
            current_q = torch.gather(current_q, dim=2, index=actions).squeeze(-1)  # Shape (64, 4)

            with torch.no_grad():
                # Compute target Q-values
                next_states = next_states.view(-1, 1, 84, 84)
                target_q = self.compute_target_values(next_states, rewards, dones)
                target_q = target_q.view(16, 4)

            # Calculate loss
            loss_q = self.loss_fn(current_q, target_q)

            # Optimize the Q-network
            self.optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()

    def memorize(self, state, action, reward, next_state, done):
        # Ensure state and next_state are in tensor format and on CPU as numpy arrays
        state = torch.tensor(state).cpu().numpy()  # (1, 84, 84)
        next_state = torch.tensor(next_state).cpu().numpy()  # (1, 84, 84)

        # Add the grayscale channel dimension if needed
        state = np.expand_dims(state, axis=0)  # (1, 1, 84, 84)
        next_state = np.expand_dims(next_state, axis=0)  # (1, 1, 84, 84)

        # Save the states in the replay buffer
        self.sequence_buffer.append((state, action, reward, next_state, done))
        if len(self.sequence_buffer) == self.chunk_size:
            self.replay_memory.append(list(self.sequence_buffer))

    def train_episode_finetuning(self, save_index):
        full_reward = 0
        state = self.env.reset()
        previnfo = None

        for _ in range(131072):  # Self.options.steps
            # Choose action with exploration
            chosen_action_id = self.epsilon_greedy(state)
            chosen_action = convertActBack(chosen_action_id)

            next_state, reward, done, info = self.env.step(chosen_action)
            if previnfo is not None:
                reward = self.myreward(info, previnfo)
            else:
                reward = 0
            full_reward += reward

            # Store in replay buffer and learn from experience
            self.memorize(state, chosen_action_id, reward, next_state, done)
            self.replay()
            # self.env.render()

            if done:
                break

            state = next_state
            self.n_steps += 1
            if self.n_steps % 50000 == 0:
                self.update_target_model()
                print("Update")

            previnfo = info
        total_reward.append(full_reward)
        # Decay epsilon after each episode
        self.decay_epsilon()
        print(self.epsilon)

    def __str__(self):
        return "DQN"


if torch.cuda.is_available():
    print("GPU")
    torch.set_default_device('cuda')


files = ['a', 'b', 'c', 'd', '1', '2', '3', '4', 'f1', 'f2', 'f3', 'f4', 'f5', 'g1', 'g2', 'g3', 'g4', 'p1', 'p2', 'p3', 'p4', 'p5', 'x', 'y', 'z', 'l3', 'l4']
path = 'C:/Users/STJoh/Documents/ReinforcementProject/Retro-Learner/' + 'a' + '.bk2'

# Initialize movie and environment once
movie = retro.Movie(path)
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

# Optionally load the model if you have pre-saved weights
if os.path.exists('mario_downsized2.pth'):
    dqn.load_model('mario_downsized2.pth')

# Main training loop across episodes
for i in range(90000):
    print(f"Starting episode {i}")
    env.initial_state = movie.get_state()
    dqn.env.reset()
    # Train for one episode
    dqn.train_episode_finetuning(i)

    # Save model after each episode or as desired
    dqn.save_model('mario_downsized2.pth')

    if i % 1000 == 0:
        # Plot rewards
        plt.figure(figsize=(10, 6))
        plt.plot(total_reward, label='Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.legend()
        plt.grid()
        plt.savefig('episode_rewards.png')
        plt.close()

# Close the environment when training is done
env.close()
