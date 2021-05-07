import os
import numpy as np

from reinforcement_learning import logger
from reinforcement_learning.gym.envs.classic_control.pendulum import PendulumEnv
from reinforcement_learning.gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from reinforcement_learning.gym.envs.box2d.bipedal_walker import BipedalWalker
from reinforcement_learning.common.vec_env import SubprocVecEnv
from reinforcement_learning.common.policies import MlpPolicy as ppo_mlp_policy
from reinforcement_learning.ddpg.policies import MlpPolicy as ddpg_mlp_policy
from reinforcement_learning.ppo2.ppo2 import PPO2 as ppo
from reinforcement_learning.ddpg.ddpg import DDPG as ddpg
from reinforcement_learning.common.callbacks import CheckpointCallback

def make_env(env_class):
    fn = lambda: env_class()
    return fn

def generate_traj(env, model, nsteps):
    n = len(env.remotes)
    states = [[] for _ in range(n)]
    actions = [[] for _ in range(n)]
    obs = env.reset()
    for i in range(nsteps):
        action, state = model.predict(obs)
        next_obs, reward, done, info = env.step(action)
        for e in range(n):
            states[e].append(obs[e])
            actions[e].append(action[e])
        obs = np.array(next_obs)
    return states, actions

if __name__ == '__main__':

    nenvs = 1
    nsteps = 1000000
    total_steps = 1000000
    env_class = BipedalWalker
    algorithm = ddpg # ppo
    policy = ddpg_mlp_policy

    modeldir = '{0}/{1}/{2}'.format('models', env_class.__name__, algorithm.__name__)
    logdir = '{0}/{1}/{2}'.format('results', env_class.__name__, algorithm.__name__)
    format_strs = os.getenv('', 'stdout,log,csv').split(',')
    logger.configure(os.path.abspath(logdir), format_strs)

    env_fns = [make_env(env_class) for _ in range(nenvs)]
    env = SubprocVecEnv(env_fns)

    if algorithm.__name__ == 'DDPG':
        eval_env_fns = [make_env(env_class)]
        eval_env = SubprocVecEnv(eval_env_fns)
        model = algorithm(policy, env, eval_env=eval_env, verbose=1)
    else:
        model = algorithm(policy, env, verbose=1)
    cb = CheckpointCallback(nsteps, modeldir, verbose=1)
    model.learn(total_timesteps=total_steps, callback=cb)