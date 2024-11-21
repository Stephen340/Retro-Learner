import retro
from time import sleep
import numpy as np
# print(retro.data.list_games())

def convertAct(action):
    if action[1] or action[8]: # There are three jump buttons; capture all in action[0]
        action[0] = 1

    if action[5]: # Down is pressed
        if action[0]:
            return 6 # Down, Up
        elif action[7]:
            return 4 # Down, Right
        elif action[6]:
            return 3 # Left, Down
        else:
            return 5 # Down
    elif action[7]:
        return 2 # right
    elif action[6]:
        return 1 # left
    elif action[0]:
        return 7 # Jump
    else:
        return 0 # no action

action_switch = {
            0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # No Operation
            1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # Left
            2: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # Right
            3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # Left, Down
            4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # Right, Down
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # Down
            6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # Down, B
            7: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # B
}
def convertActBack(actionID):
    return action_switch[actionID]

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

def main():
    # env = retro.make(game='SuperMarioBros-Nes', record='.')
    env = retro.make(game='SonicTheHedgehog-Genesis')
    obs = env.reset()
    while True:

        arr = action_switch[np.random.randint(0, 8)]
        obs, rew, done, info = env.step(arr)
        ## If the player_state is 6 the player is dead, if 11 the player is dying (in animation)
        # print(info['xscrollHi'] * 256 + info['xscrollLo'] + info['x_position'], info['y_position'])
        # print(info['player_state'])
        # x_position = xscrollHi * 256 + xscrollLo + x_position
        print(info['x'], info['y'])
        env.render()
        # sleep(0.1)
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()