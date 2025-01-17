import retro
from time import sleep
import numpy as np
# print(retro.data.list_games())


def main():
    env = retro.make(game='SuperMarioBros-Nes')
    # env = retro.make(game='SonicTheHedgehog2-Genesis')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]))
        ## If the player_state is 6 the player is dead, if 11 the player is dying (in animation)
        print(info['xscrollHi'] * 256 + info['xscrollLo'] + info['x_position'], info['y_position'])
        print(info['player_state'])
        # x_position = xscrollHi * 256 + xscrollLo + x_position
        env.render()
        sleep(0.1)
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()

# import torch
# import torch.nn as nn
#
# class DQfDNetwork(nn.Module):
#     def __init__(self, input_shape, n_actions):
#         super(DQfDNetwork, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(self.feature_size(input_shape), 512),
#             nn.ReLU(),
#             nn.Linear(512, n_actions)
#         )
#
#     def feature_size(self, input_shape):
#         return self.conv(torch.zeros(1, *input_shape)).view(1, -1).size(1)
#
#     def forward(self, x):
#         conv_out = self.conv(x).view(x.size()[0], -1)
#         return self.fc(conv_out)
#
#
#
# def train(agent, env, demo_replay_buffer, max_steps=10000):
#     state = env.reset()
#     for step in range(max_steps):
#         action = agent.select_action(state)
#         next_state, reward, done, _ = env.step(action)
#
#         # Store transition in the replay buffer
#         agent.replay_buffer.add(state, action, reward, next_state, done)
#
#         # Sample from both demo and self-play experience for training
#         demo_batch = demo_replay_buffer.sample(batch_size=32)
#         play_batch = agent.replay_buffer.sample(batch_size=32)
#         loss = agent.compute_loss(demo_batch, play_batch)
#
#         # Backpropagate the loss and update model
#         agent.optimizer.zero_grad()
#         loss.backward()
#         agent.optimizer.step()
#
#         if done:
#             state = env.reset()
#         else:
#             state = next_state
