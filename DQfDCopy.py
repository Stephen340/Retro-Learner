import random
from copy import deepcopy
from collections import deque
import retro
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import sleep
from torch.optim import AdamW


class QFunction(nn.Module):
    """
    Q-network definition.
    """

    def __init__(
            self,
            obs_dim,
            act_dim,
            hidden_sizes,
    ):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(dim=-1)


class DQN():
    def __init__(self, env, eval_env, options):
        assert str(env.action_space).startswith("Discrete") or str(
            env.action_space
        ).startswith("Tuple(Discrete"), (
                str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)
        # Create Q-network
        self.model = QFunction(
            env.observation_space.shape[0],
            env.action_space.n,
            [64, 64],
        )
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
        self.replay_memory = deque(maxlen=options.replay_memory_size)

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
        """
        Apply an epsilon-greedy policy based on the given Q-function approximator and epsilon.

        Returns:
            The probabilities (as a Numpy array) associated with each action for 'state'.

        Use:
            self.env.action_space.n: Number of avilable actions
            self.torch.as_tensor(state): Convert Numpy array ('state') to a tensor
            self.model(state): Returns the predicted Q values at a
                'state' as a tensor. One value per action.
            torch.argmax(values): Returns the index corresponding to the highest value in
                'values' (a tensor)
        """
        nA = self.env.action_space.n
        # Convert to tensor and get the predicted Q values
        state = torch.as_tensor(state)
        action_values = self.model(state)
        # Get the greedy action
        greedy_action = torch.argmax(action_values)
        # List of probs, initialized as the eps/len(AcSpace)
        probability_per_action = np.ones(nA) * (0.90 / nA)  # .90 is greedy epsilon. Chance of random exploration
        # Update the greedy action appropriately
        probability_per_action[greedy_action] = 1.0 - .90 + .90 / nA
        return probability_per_action

    def compute_target_values(self, next_states, rewards, dones):
        """
        Computes the target q values.

        Returns:
            The target q value (as a tensor) of shape [len(next_states)]
        """
        next_q_vals = self.target_model(next_states)
        best_next_q_vals = torch.max(next_q_vals, dim=1)[0]  # Get max qs for each state
        target = rewards + 0.9 * best_next_q_vals * (1 - dones)  # self.options.gamma is 0.9
        return torch.as_tensor(target)

    def myreward(self, info, previnfo):
        # NOTE: Make sure you update data.json or x_position2 will not be found

        # previous
        pxpos = previnfo['x_position2'] + previnfo['xscrollLo'] + 256 * previnfo['xscrollHi']
        ptime = previnfo['time']

        # now
        xpos = info['x_position2'] + info['xscrollLo'] + 256 * info['xscrollHi']
        isDead = info['player_state'] != 8
        time = info['time']

        # change
        dpos = xpos - pxpos
        dtime = time - ptime

        return (-15 if isDead else 0) + dpos + 0.1 * dtime

    def replay(self):
        """
        TD learning for q values on past transitions.

        Use:
            self.target_model(state): predicted q values as an array with entry
                per action
        """
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
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        """
        Perform a single episode of the Q-Learning algorithm for off-policy TD
        control using a DNN Function Approximation. Finds the optimal greedy policy
        while following an epsilon-greedy policy.

        Use:
            self.epsilon_greedy(state): return probabilities of actions.
            np.random.choice(array, p=prob): sample an element from 'array' based on their corresponding
                probabilites 'prob'.
            self.memorize(state, action, reward, next_state, done): store the transition in the replay buffer
            self.update_target_model(): copy weights from model to target_model
            self.replay(): TD learning for q values on past transitions
            self.options.update_target_estimator_every: Copy parameters from the Q estimator to the
                target estimator every N steps (HINT: to be done across episodes)
        """

        # Reset the environment
        state, _ = self.env.reset()
        testing = False

        for _ in range(131072):  # Self.options.steps
            # Get action
            if testing:
                probabilities = self.epsilon_greedy(state)
                chosen_action = np.random.choice(np.arange(len(probabilities)), p=probabilities)
                next_state, reward, done, _ = env.step(chosen_action)
            else:
                movie = retro.Movie('C:/Users/stjoh/Documents/CSCE 642/d.bk2')
                movie.step()

                env = retro.make(
                    game=movie.get_game(),
                    state=None,
                    # bk2s can contain any button presses, so allow everything
                    use_restricted_actions=retro.Actions.ALL,
                    players=movie.players,
                )
                env.initial_state = movie.get_state()
                state = movie.get_state()
                next_state, reward, done, info, chosen_action = None, None, None, None, None
                previnfo = None
                while movie.step():
                    chosen_action = []
                    for p in range(movie.players):
                        for i in range(env.num_buttons):
                            chosen_action.append(movie.get_key(i, p))
                    # Note: The action space is stored in the keys variable itself.
                    # The keys is an array of 9 booleans, each mapping to an action (move right, jump, etc.)
                    # True = Key is pressed and action is being used; False otherwise
                    next_state, reward, done, info = env.step(chosen_action)
                    env.render()
                    # print(info['x_position2'] + info['xscrollLo'] + 256 * info['xscrollHi'])
                    if previnfo is not None:
                        print(self.myreward(info, previnfo))
                        reward = self.myreward(info, previnfo)
                    else:
                        reward = 0
                    sleep(0.01)
                    previnfo = info

            # Memorize and replay
            self.memorize(state, chosen_action, reward, next_state, done)
            self.replay()
            if done:
                break
            # Set state, update target model if needed, increment step count
            state = next_state
            if self.n_steps % 100 == 0:
                self.update_target_model()
            self.n_steps += 1

    def __str__(self):
        return "DQN"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.


        Returns:
            A function that takes an observation as input and returns a greedy
            action
        """

        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            q_values = self.model(state)
            return torch.argmax(q_values).detach().numpy()

        return policy_fn
