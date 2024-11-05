import retro
import numpy as np
from time import sleep

# print(retro.data.list_games())

import torch
import torch.nn as nn

class DQfDNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQfDNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def feature_size(self, input_shape):
        return self.conv(torch.zeros(1, *input_shape)).view(1, -1).size(1)

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# STATE SPACE: 224 x 240 x 3
# ACTION SIZE: 9
def train(agent, env, demo_replay_buffer, max_steps=10000):
    state = env.reset()
    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        # TODO: Parse state into grayscale & other properties if needed
        #      - state is simply the 3d array of pixels (224 x 240 x 3)

        # TODO: Define a proper reward
        #      - positive: (info.leftShift or smth like that)
        #      - negative: death

        # TODO: Add some sort of replay buffer
        #       (in the future can add to https://retro.readthedocs.io/en/latest/python.html#replay-files)

        # Store transition in the replay buffer
        agent.replay_buffer.add(state, action, reward, next_state, done)

        # TODO: Update the deep neural network within the agent

        # Sample from both demo and self-play experience for training
        demo_batch = demo_replay_buffer.sample(batch_size=32)
        play_batch = agent.replay_buffer.sample(batch_size=32)
        loss = agent.compute_loss(demo_batch, play_batch)

        # Backpropagate the loss and update model
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        if done:
            state = env.reset()
        else:
            state = next_state

def main():
    env = retro.make(game='SuperMarioBros-Nes', record='D:/')
    obs = env.reset()
    lastLivesNum = 3
    previnfo = None
    while True:
        # obs, rew, done, info = env.step(env.action_space.sample())
        obs, rew, done, info = env.step(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]))
        if previnfo is not None:
            print(myreward(info, previnfo))

        env.render()
        sleep(0.01)
        if info['lives'] < lastLivesNum:
            print("Death")
            lastLivesNum = info['lives']
        if done:
            obs = env.reset()
            break
        previnfo = info
    env.close()

def myreward(info, previnfo):
    # NOTE FOR STEVEN: Make sure you update data.json or x_position2 will not be found

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


# main2: replay the file
def main2():
    # TODO: set the file path to match where you recorded the file
    movie = retro.Movie('C:\\Users\\sturt\\Desktop\\replays\\d.bk2')
    movie.step()

    env = retro.make(
        game=movie.get_game(),
        state=None,
        # bk2s can contain any button presses, so allow everything
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
    )
    env.initial_state = movie.get_state()
    env.reset()

    previnfo = None
    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))
        # Note: The action space is stored in the keys variable itself.
        # The keys is an array of 9 booleans, each mapping to an action (move right, jump, etc.)
        # True = Key is pressed and action is being used; False otherwise
        obs, rew, done, info = env.step(keys)
        env.render()
        # print(info['x_position2'] + info['xscrollLo'] + 256 * info['xscrollHi'])
        if previnfo is not None:
            print(myreward(info, previnfo))
        sleep(0.01)
        previnfo = info

if __name__ == "__main__":
    # main() use agent
    main()
    # main()

