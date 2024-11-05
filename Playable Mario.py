import retro
from time import sleep
import numpy as np
# print(retro.data.list_games())


def main():
    # env = retro.make(game='SuperMarioBros-Nes', record='.')
    env = retro.make(game='SonicTheHedgehog-Genesis')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]))
        ## If the player_state is 6 the player is dead, if 11 the player is dying (in animation)
        # print(info['xscrollHi'] * 256 + info['xscrollLo'] + info['x_position'], info['y_position'])
        # print(info['player_state'])
        # x_position = xscrollHi * 256 + xscrollLo + x_position
        env.render()
        # sleep(0.1)
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()