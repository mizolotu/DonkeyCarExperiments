import argparse as arp
import os
import numpy as np

from reinforcement_learning import logger
from reinforcement_learning.gym.envs.classic_control.pendulum import PendulumEnv
from reinforcement_learning.gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from reinforcement_learning.gym.envs.box2d.lunar_lander import LunarLanderContinuous
from reinforcement_learning.gym.envs.box2d.bipedal_walker import BipedalWalker
from reinforcement_learning.common.callbacks import CheckpointCallback

from reinforcement_learning.common.vec_env.subproc_vec_env import SubprocVecEnv
from reinforcement_learning.ddpg.policies import MlpPolicy as ddpg_policy
from reinforcement_learning.sac.policies import MlpPolicy as sac_policy

from reinforcement_learning.ppo2.ppo2 import PPO2 as ppo
from reinforcement_learning.ddpg.ddpg import DDPG as ddpg
from reinforcement_learning.sac.sac import SAC as sac

from on_policy_experiments import make_env, generate_traj, find_checkpoint_with_highest_explained_variance

env_list = [
    PendulumEnv,
    Continuous_MountainCarEnv,
    BipedalWalker,
    LunarLanderContinuous
]

algorithm_list = [
    ddpg,
    sac
]

policy_list = [
    ddpg_policy,
    sac_policy
]

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Test state-of-art RL alghorithms in OpenAI gym')
    parser.add_argument('-e', '--env', help='Environment index', type=int, default=0)
    parser.add_argument('-s', '--steps', help='Number of episode steps', type=int, default=64)
    parser.add_argument('-u', '--updates', help='Number of updates', type=int, default=40000)
    parser.add_argument('-a', '--algorithm', help='RL algorithm index', type=int, default=1)
    parser.add_argument('-o', '--output', help='Output directory', default='models')
    parser.add_argument('-c', '--cuda', help='Use CUDA', default=False, type=bool)
    parser.add_argument('-t', '--trainer', help='Expert model', default='PPO2/policy_1_expert')
    parser.add_argument('-p', '--pretrain', help='Full pretrain', default=False, type=bool)
    args = parser.parse_args()

    if not args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    env_class = env_list[args.env]
    algorithm = algorithm_list[args.algorithm]
    policy = policy_list[args.algorithm]
    totalsteps = args.steps * args.updates
    env_fns = [make_env(env_class)]
    env = SubprocVecEnv(env_fns)
    eval_env_fns = [make_env(env_class) for _ in range(1)]
    eval_env = SubprocVecEnv(eval_env_fns)

    if args.trainer is not None:
        if args.pretrain:
            postfix = 'ac'
        else:
            postfix = 'bc'
        #checkpoint_file = f'{args.output}/{env_class.__name__}/{args.trainer}/rl_model_{good_checkpoints[args.env]}_steps.zip'
        checkpoint_file = find_checkpoint_with_highest_explained_variance(f'{args.output}/{env_class.__name__}/{args.trainer}')
        trainer_model = ppo.load(checkpoint_file)
        trainer_model.set_env(env)
        print('Expert model has been successfully loaded from {0}'.format(checkpoint_file))

        trajs = []
        for i in range(100000 // args.steps):
            states, actions, next_states, rewards = generate_traj(env, trainer_model, args.steps)
            for se, ae, ne, re in zip(states, actions, next_states, rewards):
                trajs.append([])
                for s, a, n, r in zip(se, ae, ne, re):
                    trajs[-1].append(np.hstack([s, a, n, r]))
                trajs[-1] = np.vstack(trajs[-1])
        trajs = np.vstack(trajs)

        del trainer_model
    else:
        postfix = 'pure'

    logdir = f'{args.output}/{env_class.__name__}/{algorithm.__name__}/{policy.__name__}_{postfix}/'
    format_strs = os.getenv('', 'stdout,log,csv').split(',')
    logger.configure(os.path.abspath(logdir), format_strs)

    model = algorithm(policy, env, eval_env=eval_env, n_steps=args.steps, verbose=1)
    if postfix == 'bc':
        model.pretrain(trajs, batch_size=args.steps, n_epochs=10, learning_rate=1e-3)
    elif postfix == 'ac':
        model.full_pretrain(trajs, batch_size=args.steps, n_epochs=1)
        print(len(model.replay_buffer))

    cb = CheckpointCallback(args.steps * args.updates, logdir, verbose=1)
    model.learn(total_timesteps=totalsteps, callback=cb)



