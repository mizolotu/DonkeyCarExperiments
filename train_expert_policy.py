import argparse as arp
import os, pandas
import numpy as np

from reinforcement_learning import logger
from reinforcement_learning.gym.envs.classic_control.pendulum import PendulumEnv
from reinforcement_learning.gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from reinforcement_learning.gym.envs.box2d.lunar_lander import LunarLanderContinuous
from reinforcement_learning.gym.envs.box2d.bipedal_walker import BipedalWalker
from reinforcement_learning.common.callbacks import CheckpointCallback
from reinforcement_learning.common.vec_env.subproc_vec_env import SubprocVecEnv
from reinforcement_learning.common.policies import policy_1 as policy
from reinforcement_learning.ppo2.ppo2 import PPO2 as ppo

from on_policy_experiments import make_env, env_list, generate_traj

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Create an expert policy')
    parser.add_argument('-e', '--env', help='Environment index', type=int, default=0)
    parser.add_argument('-n', '--nenvs', help='Number of environments', type=int, default=16)
    parser.add_argument('-s', '--steps', help='Number of episode steps', type=int, default=64)
    parser.add_argument('-u', '--updates', help='Number of updates', type=int, default=1000)
    parser.add_argument('-o', '--output', help='Output directory', default='models')
    parser.add_argument('-c', '--cuda', help='Use CUDA', default=False, type=bool)
    args = parser.parse_args()

    if not args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    env_class = env_list[args.env]
    nenvs = args.nenvs
    algorithm = ppo
    totalsteps = args.steps * args.updates * nenvs
    env_fns = [make_env(env_class) for _ in range(nenvs)]
    env = SubprocVecEnv(env_fns)

    postfix = 'expert'
    logdir = f'{args.output}/{env_class.__name__}/{algorithm.__name__}/{policy.__name__}_{postfix}/'
    format_strs = os.getenv('', 'stdout,log,csv').split(',')
    logger.configure(os.path.abspath(logdir), format_strs)
    model = algorithm(policy, env, n_steps=args.steps, verbose=1)
    cb = CheckpointCallback(args.steps * nenvs, logdir, verbose=1)
    model.learn(total_timesteps=totalsteps, callback=cb)