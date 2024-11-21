import gc
import os
import random
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

rewards_per_episode = []

def convertAct(action):
    if action[1] or action[8]:  # There are three jump buttons; capture all in action[0]
        action[0] = 1

    if action[5]:  # Down is pressed
        if action[0]:
            return 6  # Down, Up
        elif action[7]:
            return 4  # Down, Right
        elif action[6]:
            return 3  # Left, Down
        else:
            return 5  # Down
    elif action[7]:
        return 2  # right
    elif action[6]:
        return 1  # left
    elif action[0]:
        return 7  # Jump
    else:
        return 0 # no action

action_switch = {
            0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No Operation
            1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Left
            2: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Right
            3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Left, Down
            4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # Right, Down
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Down
            6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Down, B
            7: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # B
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


N_ACTIONS = 12


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

        if os.path.exists('sonic_pretrained.pth'):
            self.load_model('sonic_pretrained.pth')
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

    def compute_target_values(self, next_states, rewards, dones):
        next_q_vals = self.target_model(next_states)  # Shape should be (64 * 4, num_actions)
        best_next_q_vals = torch.max(next_q_vals, dim=1)[0]  # Shape should be (64 * 4)
        best_next_q_vals = best_next_q_vals.view(16, 4)  # Shape (64, 4)
        target = rewards + 0.9 * best_next_q_vals * (1 - dones)  # 0.9 is gamma

        return target

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

    def train_episode(self):
        state = self.env.reset()

        for _ in range(131072):  # Self.options.steps
            # If a movie is loaded, step through the movie instead
            if not movie.step():  # Movie replay has ended
                break

            # derive the actions from the pressed keys
            chosen_action = []
            for p in range(movie.players):
                for i in range(env.num_buttons):
                    chosen_action.append(movie.get_key(i, p))
            chosen_action_id = convertAct(chosen_action)

            # step through the environment with the chosen action
            next_state, reward, done, info = self.env.step(chosen_action)
            # print(reward)

            # update replay memory & model
            self.memorize(state, chosen_action_id, reward, next_state, done)
            self.replay()
            # self.env.render()
            if done:
                break

            # Update variables for next step
            state = next_state
            self.n_steps += 1
            if self.n_steps % 10000 == 0:
                print("UPDATE")
                self.update_target_model()

        # self.save_model("sonic_pretrained.pth")

    def __str__(self):
        return "DQN"


if torch.cuda.is_available():
    torch.set_default_device('cuda')

files = ['sa', 'sb', 'sc', 'sd', 'se', 'sf', 'sg', 'sh', 'si', 'sj', 'sk', 'sl', 'sm', 'sn', 'so']
movie = retro.Movie('C:/Users/stjoh/Documents/CSCE 642/SonicRecordings/sa.bk2')
movie.step()

env = retro.make(
    game=movie.get_game(),
    state=None,
    use_restricted_actions=retro.Actions.ALL,
    players=movie.players,
)
env = FrameSkip(env, 4)
env = ResizeObservation(env, 84)

dqn = DQN(env, movie)  # Initialize the DQN model once
movie.close()
env.render(close=True)
env.close()
del movie
del env
gc.collect()

for i in range(1000):
    print(f"Episode {i + 1}")
    path = f'C:/Users/stjoh/Documents/CSCE 642/SonicRecordings/{files[i % len(files)]}.bk2'
    movie = retro.Movie(path)
    movie.step()

    # Update environment with new movie
    env = retro.make(
        game=movie.get_game(),
        state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
    )
    env.initial_state = movie.get_state()
    env.reset()
    # env = FrameSkip(env, 4)
    env = ResizeObservation(env, 84)

    # Train the DQN on the current movie
    dqn.env = env  # Update the environment in the existing DQN
    dqn.movie = movie  # Update the movie in the existing DQN
    dqn.train_episode()

    # Clean up for this movie
    env.render(close=True)
    movie.close()
    env.close()
    del movie
    del env
    gc.collect()

    dqn.save_model("sonic_pretrained.pth")

# Plot rewards
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards Per Episode')
plt.savefig('rewards_per_episode.png')
plt.show()
import gc
import os
import random
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

rewards_per_episode = []

def convertAct(action):
    if action[1] or action[8]:  # There are three jump buttons; capture all in action[0]
        action[0] = 1

    if action[5]:  # Down is pressed
        if action[0]:
            return 6  # Down, Up
        elif action[7]:
            return 4  # Down, Right
        elif action[6]:
            return 3  # Left, Down
        else:
            return 5  # Down
    elif action[7]:
        return 2  # right
    elif action[6]:
        return 1  # left
    elif action[0]:
        return 7  # Jump
    else:
        return 0 # no action

action_switch = {
            0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No Operation
            1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Left
            2: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Right
            3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Left, Down
            4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # Right, Down
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Down
            6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Down, B
            7: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # B
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


N_ACTIONS = 12


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

        if os.path.exists('sonic_pretrained.pth'):
            self.load_model('sonic_pretrained.pth')
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

    def compute_target_values(self, next_states, rewards, dones):
        next_q_vals = self.target_model(next_states)  # Shape should be (64 * 4, num_actions)
        best_next_q_vals = torch.max(next_q_vals, dim=1)[0]  # Shape should be (64 * 4)
        best_next_q_vals = best_next_q_vals.view(16, 4)  # Shape (64, 4)
        target = rewards + 0.9 * best_next_q_vals * (1 - dones)  # 0.9 is gamma

        return target

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

    def myreward(self, info, previnfo):
        # Calculate positions
        pxpos = previnfo['x']
        xpos = info['x']

        # Check status flags
        isDead = info['lives'] < 3
        if isDead:
            return -100
        isFlag = info['level_end_bonus'] > 0
        if isFlag:
            return 2000  # Reward for reaching the flag

        # Calculate changes
        dpos = xpos - pxpos  # Change in x position
        dtime = -1  # Change in time

        # Reward components
        rightward_reward = dpos * 2 if dpos > 0 else -5  # 2x reward for moving right, -5 for left
        time_bonus = 0.05 * dtime  # Smaller time reward

        # Final reward calculation
        reward = rightward_reward + time_bonus
        return reward

    def train_episode(self):
        state = self.env.reset()
        prev_info = None

        for _ in range(131072):  # Self.options.steps
            # If a movie is loaded, step through the movie instead
            if not movie.step():  # Movie replay has ended
                break

            # derive the actions from the pressed keys
            chosen_action = []
            for p in range(movie.players):
                for i in range(env.num_buttons):
                    chosen_action.append(movie.get_key(i, p))
            chosen_action_id = convertAct(chosen_action)

            # step through the environment with the chosen action
            next_state, reward, done, info = self.env.step(chosen_action)
            # print(reward)
            if prev_info is not None:
                reward = self.myreward(info, prev_info)
            prev_info = info

            # update replay memory & model
            self.memorize(state, chosen_action_id, reward, next_state, done)
            self.replay()
            # self.env.render()
            if done:
                break

            # Update variables for next step
            state = next_state
            self.n_steps += 1
            if self.n_steps % 10000 == 0:
                print("UPDATE")
                self.update_target_model()

        # self.save_model("sonic_pretrained.pth")

    def __str__(self):
        return "DQN"


if torch.cuda.is_available():
    torch.set_default_device('cuda')

files = ['sa', 'sb', 'sc', 'sd', 'se', 'sf', 'sg', 'sh', 'si', 'sj', 'sk', 'sl', 'sm', 'sn', 'so']
movie = retro.Movie('C:/Users/stjoh/Documents/CSCE 642/SonicRecordings/sa.bk2')
movie.step()

env = retro.make(
    game=movie.get_game(),
    state=None,
    use_restricted_actions=retro.Actions.ALL,
    players=movie.players,
)
env = FrameSkip(env, 4)
env = ResizeObservation(env, 84)

dqn = DQN(env, movie)  # Initialize the DQN model once
movie.close()
env.render(close=True)
env.close()
del movie
del env
gc.collect()

for i in range(1000):
    print(f"Episode {i + 1}")
    path = f'C:/Users/stjoh/Documents/CSCE 642/SonicRecordings/{files[i % len(files)]}.bk2'
    movie = retro.Movie(path)
    movie.step()

    # Update environment with new movie
    env = retro.make(
        game=movie.get_game(),
        state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
    )
    env.initial_state = movie.get_state()
    env.reset()
    # env = FrameSkip(env, 4)
    env = ResizeObservation(env, 84)

    # Train the DQN on the current movie
    dqn.env = env  # Update the environment in the existing DQN
    dqn.movie = movie  # Update the movie in the existing DQN
    dqn.train_episode()

    # Clean up for this movie
    env.render(close=True)
    movie.close()
    env.close()
    del movie
    del env
    gc.collect()

    dqn.save_model("sonic_pretrained.pth")

# Plot rewards
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards Per Episode')
plt.savefig('rewards_per_episode.png')
plt.show()