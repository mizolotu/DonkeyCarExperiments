import os
import gym_donkeycar
import numpy as np

if __name__ == '__main__':

    exe_path = '/home/mizolotu/DonkeyCar/donkey_sim.x86_64'
    port = 9091
    conf = {'exe_path': exe_path, 'port' : port}
    env = gym.make("donkey-generated-track-v0", conf=conf)

    obs = env.reset()
    for t in range(100):
        action = np.array([0.0, 0.5]) # drive straight with small speed
        obs, reward, done, info = env.step(action)
    env.close()