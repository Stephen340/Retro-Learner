import os
import random
from collections import deque
from copy import deepcopy

import numpy as np
import retro
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim import AdamW


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
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 9, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
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

        if os.path.exists('mario_new_data.pth'):
            self.load_model('mario_new_data.pth')
        self.model.to('cuda')
        self.target_model.to('cuda')

        # Replay buffer
        self.replay_memory = deque(maxlen=1000000)  # Max replay size

        # Number of training steps so far
        self.n_steps = 0
        self.epsilon = 0.  # Start with full exploration
        self.epsilon_min = 0.8  # Minimum exploration
        self.epsilon_decay = 0.995  # Decay rate
        self.leftward_counter = 0  # Each frame with left movement

    def handle_episodes(self):
        for i in range(5000):
            print(i)
            dqn.train_episode_finetuning(i)
            if i % 10 == 9:
                dqn.save_model('mario_rightward.pth')
            env.reset()

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
        grayscale_transform = transforms.Grayscale()
        state = grayscale_transform(torch.tensor(state).permute(2, 0, 1)).unsqueeze(0).float()
        action_values = self.model(state)
        return torch.argmax(action_values).item()

    def epsilon_greedy(self, state):
        nA = N_ACTIONS
        grayscale_transform = transforms.Grayscale()
        state = grayscale_transform(torch.tensor(state).permute(2, 0, 1)).unsqueeze(0).float()
        action_values = self.model(state)

        # Exploration vs. Exploitation choice
        if random.random() < self.epsilon:  # 0.80 is self.epsilon
            # Choose a random action
            chosen_action = np.random.choice(np.arange(nA))
        else:
            # Choose the best action based on model predictions
            chosen_action = torch.argmax(action_values).item()

        return chosen_action

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            print(self.epsilon * self.epsilon_decay)
            self.epsilon *= self.epsilon_decay

    def compute_target_values(self, next_states, rewards, dones):
        next_q_vals = self.target_model(next_states)  # Shape should be (64 * 4, num_actions)
        best_next_q_vals = torch.max(next_q_vals, dim=1)[0]  # Shape should be (64 * 4)
        best_next_q_vals = best_next_q_vals.view(8, 4)  # Shape (64, 4)
        target = rewards + 0.9 * best_next_q_vals * (1 - dones)  # 0.9 is gamma

        return target

    def myreward(self, info, previnfo):
        # Calculate positions
        pxpos = previnfo['x_position2'] + previnfo['xscrollLo'] + 256 * previnfo['xscrollHi']
        xpos = info['x_position2'] + info['xscrollLo'] + 256 * info['xscrollHi']

        # Check status flags
        isDead = info['player_state'] == 6 or info['player_state'] == 11
        if isDead:
            return -100
        isFlag = info['player_state'] == 4
        if isFlag:
            return 2000  # Reward for reaching the flag

        # Calculate changes
        dpos = xpos - pxpos  # Change in x position
        dtime = info['time'] - previnfo['time']  # Change in time

        # Reward components
        rightward_reward = dpos * 2 if dpos > 0 else -5  # 2x reward for moving right, -5 for left
        time_bonus = 0.05 * dtime  # Smaller time reward

        # Final reward calculation
        reward = rightward_reward + time_bonus
        return reward

    def replay(self):
        if len(self.replay_memory) > 8:  # 64 is self.options.batch_size
            minibatch = random.sample(self.replay_memory, 8)
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
            states = states.view(-1, 1, 224, 240)
            next_states = next_states.view(-1, 1, 224, 240)

            # Calculate current Q-values
            current_q = self.model(states)
            current_q = current_q.view(8, 4, -1)  # Reshape back to (batch_size, chunk_size, num_actions)

            # Gather Q-values for actions taken in replay memory
            actions = actions.unsqueeze(-1)  # Shape (64, 4, 1) to match current_q for gather
            current_q = torch.gather(current_q, dim=2, index=actions).squeeze(-1)  # Shape (64, 4)

            with torch.no_grad():
                # Compute target Q-values
                next_states = next_states.view(-1, 1, 224, 240)
                target_q = self.compute_target_values(next_states, rewards, dones)
                target_q = target_q.view(8, 4)

            # Calculate loss
            loss_q = self.loss_fn(current_q, target_q)

            # Optimize the Q-network
            self.optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()

    def memorize(self, state, action, reward, next_state, done):
        # print(state.shape)
        grayscale_transform = transforms.Grayscale()
        state = grayscale_transform(torch.tensor(state).permute(2, 0, 1)).cpu().numpy()  # (224, 240)
        next_state = grayscale_transform(torch.tensor(next_state).permute(2, 0, 1)).cpu().numpy()  # (224, 240)
        state = np.expand_dims(state, axis=0)  # Now (1, 224, 240)
        next_state = np.expand_dims(next_state, axis=0)  # Now (1, 224, 240)

        # Save the grayscale states in the buffer
        self.sequence_buffer.append((state, action, reward, next_state, done))
        if len(self.sequence_buffer) == self.chunk_size:
            self.replay_memory.append(list(self.sequence_buffer))

    def train_episode_finetuning(self, save_index):
        state = self.env.reset()
        previnfo = None
        max_loc = -10

        for _ in range(131072):  # Self.options.steps
            # Choose action with exploration
            chosen_action_id = self.epsilon_greedy(state)
            chosen_action = convertActBack(chosen_action_id)

            next_state, reward, done, info = self.env.step(chosen_action)
            if previnfo is not None:
                reward = self.myreward(info, previnfo)
            else:
                reward = 0

            if reward == -100:
                done |= True

            if (info['x_position2'] + info['xscrollLo'] + 256 * info['xscrollHi']) > max_loc:
                max_loc = info['x_position2'] + info['xscrollLo'] + 256 * info['xscrollHi']

            # Store in replay buffer and learn from experience
            self.memorize(state, chosen_action_id, reward, next_state, done)
            self.replay()
            self.env.render()

            if done:
                self.update_target_model()
                break

            state = next_state
            self.n_steps += 1
            if self.n_steps % 1000 == 0:
                self.update_target_model()

            previnfo = info

        # Decay epsilon after each episode
        self.decay_epsilon()
        self.save_model("mario_new_data.pth")

    def __str__(self):
        return "DQN"


if torch.cuda.is_available():
    torch.set_default_device('cuda')

files = ['a', 'b', 'c', 'd', '1', '2', '3', '4', 'f1', 'f2', 'f3', 'f4', 'f5', 'g1', 'g2', 'g3', 'g4', 'p1', 'p2', 'p3', 'p4', 'p5', 'x', 'y', 'z', 'l3', 'l4']
path = 'C:/Users/stjoh/Documents/CSCE 642/' + 'a' + '.bk2'
movie = retro.Movie(path)
env = retro.make(
        game=movie.get_game(),
        state=None,
        # bk2s can contain any button presses, so allow everything
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
    )
dqn = DQN(env, movie)
dqn.load_model('mario_rightward.pth')
dqn.handle_episodes()
