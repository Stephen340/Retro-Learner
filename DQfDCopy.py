import os
import random
from copy import deepcopy
from collections import deque
import retro
import torch
import torch.nn as nn
import numpy as np
from time import sleep
from torch.optim import AdamW


def convertAct(action):
    if action[-1] == 1:
        if action[-2] == 1:
            return 3
        elif action[-3] == 1:
            return 5
        else:
            return 2
    else:
        if action[-2] == 1:
            return 1
        elif action[-3] == 1:
            return 4
        else:
            return 0

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
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(26 * 24 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, N_ACTIONS)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DQN:
    def __init__(self, env, movie):
        # Create CNN
        self.movie = movie
        self.env = env
        self.model = CustomCNN()  # Action space

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
        self.replay_memory = deque(maxlen=300000) # Max replay size

        # Number of training steps so far
        self.n_steps = 0

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

    def epsilon_greedy(self, state):
        nA = N_ACTIONS
        state = torch.as_tensor(state)
        action_values = self.model(state)
        greedy_action = torch.argmax(action_values)
        probability_per_action = np.ones(nA) * (0.90 / nA)  # .90 is greedy epsilon. Chance of random exploration
        # Update the greedy action appropriately
        probability_per_action[greedy_action] = 1.0 - .90 + .90 / nA
        return probability_per_action

    def compute_target_values(self, next_states, rewards, dones):
        next_q_vals = self.target_model(next_states)
        best_next_q_vals = torch.max(next_q_vals, dim=1)[0]  # Get max qs for each state
        target = rewards + 0.9 * best_next_q_vals * (1 - dones)  # self.options.gamma is 0.9
        return torch.as_tensor(target)

    def myreward(self, info, previnfo):
        # previous state
        pxpos = previnfo['x_position2'] + previnfo['xscrollLo'] + 256 * previnfo['xscrollHi']
        ptime = previnfo['time']

        # current state
        xpos = info['x_position2'] + info['xscrollLo'] + 256 * info['xscrollHi']
        isDead = info['player_state'] != 8
        time = info['time']

        # change between the two states
        dpos = xpos - pxpos
        dtime = time - ptime

        return (-15 if isDead else 0) + dpos + 0.1 * dtime

    def replay(self):
        if len(self.replay_memory) > 64:  # 64 is self.options.batch_size
            minibatch = random.sample(self.replay_memory, 64)
            minibatch = [
                np.array(
                    [
                        transition[idx]
                        for transition, idx in zip(minibatch, [i] * len(minibatch))
                    ]
                )
                for i in range(5)
            ]
            states, actions, rewards, next_states, dones = minibatch
            # Convert numpy arrays to torch tensors
            states = torch.as_tensor(states, dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.float32)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            next_states = torch.as_tensor(next_states, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)

            # Current Q-values
            if states.shape[1] != 3:
                states = np.transpose(states, (0, 3, 1, 2))
            current_q = self.model(states)
            # Q-values for actions in the replay memory
            current_q = torch.gather(
                current_q, dim=1, index=actions.unsqueeze(1).long()
            ).squeeze(-1)

            with torch.no_grad():
                target_q = self.compute_target_values(next_states, rewards, dones)

            # Calculate loss
            loss_q = self.loss_fn(current_q, target_q)

            # Optimize the Q-network
            self.optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()

    def memorize(self, state, action, reward, next_state, done):
        state = np.transpose(state, (2, 0, 1))
        next_state = np.transpose(next_state, (2, 0, 1))
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        state = self.env.reset()
        previnfo = None

        for _ in range(131072):  # Self.options.steps
            # If no movie is loaded, randomly select the next action
            if movie is None:
                probabilities = self.epsilon_greedy(state)
                chosen_action_id = np.random.choice(np.arange(len(probabilities)), p=probabilities)
                chosen_action = convertActBack(chosen_action_id)

            # If a movie is loaded, step through the movie instead
            else:
                if not movie.step(): # Movie replay has ended
                    break

                # derive the actions from the pressed keys
                chosen_action = []
                for p in range(movie.players):
                    for i in range(env.num_buttons):
                        chosen_action.append(movie.get_key(i, p))
                chosen_action_id = convertAct(chosen_action)

            # step through the environment with the chosen action
            next_state, reward, done, info = env.step(chosen_action)

            # calculate reward using previous data for mario
            if previnfo is not None:
                # print(self.myreward(info, previnfo))
                reward = self.myreward(info, previnfo)
            else:
                reward = 0

            # update replay memory & model
            self.memorize(state, chosen_action_id, reward, next_state, done)
            self.replay()
            self.env.render()
            if done:
                break

            # Update variables for next step
            state = next_state
            if self.n_steps % 1000 == 0:
                self.update_target_model()
            self.n_steps += 1

            previnfo = info

    def __str__(self):
        return "DQN"


movie = retro.Movie(os.path.join(os.path.dirname(os.getcwd()), 'd.bk2'))
movie.step()

env = retro.make(
    game=movie.get_game(),
    state=None,
    # bk2s can contain any button presses, so allow everything
    use_restricted_actions=retro.Actions.ALL,
    players=movie.players,
)
dqn = DQN(env, movie)
dqn.train_episode()
