import argparse as arp
import os
import numpy as np

from reinforcement_learning import logger
from reinforcement_learning.gym.envs.donkey_car.donkey_env import DonkeyEnv
from reinforcement_learning.common.callbacks import CheckpointCallback

from reinforcement_learning.common.vec_env.subproc_vec_env import SubprocVecEnv
from reinforcement_learning.sac.policies import MlpPolicy, CnnPolicy
from reinforcement_learning.sac.sac import SAC as sac

from on_policy_experiments import generate_traj, find_checkpoint_with_highest_explained_variance

def make_env(env_class, *args):
    fn = lambda: env_class(*args)
    return fn

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Test state-of-art RL alghorithms in OpenAI gym')
    parser.add_argument('-e', '--env', help='Environment index', type=int, default=1)
    parser.add_argument('-s', '--steps', help='Number of episode steps', type=int, default=256)
    parser.add_argument('-u', '--updates', help='Number of updates', type=int, default=10000)
    parser.add_argument('-a', '--algorithm', help='RL algorithm index', type=int, default=0)
    parser.add_argument('-o', '--output', help='Output directory', default='models')
    parser.add_argument('-c', '--cuda', help='Use CUDA', default=False, type=bool)
    parser.add_argument('-t', '--trainer', help='Expert model', default='SAC/MlpPolicy_expert')
    args = parser.parse_args()

    if not args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    env_class = DonkeyEnv
    conf = {'exe_path': '/home/mizolotu/DonkeyCar/donkey_sim.x86_64', 'port': 9091}
    level = 'generated_track'
    algorithm = sac
    policy = [CnnPolicy, MlpPolicy][args.env]
    totalsteps = args.steps * args.updates
    env_fns = [make_env(env_class, level, conf, args.env)]
    env = SubprocVecEnv(env_fns)

    logdir = f'{args.output}/{env_class.__name__}_{args.env}/{algorithm.__name__}/{policy.__name__}/'
    format_strs = os.getenv('', 'stdout,log,csv').split(',')
    logger.configure(os.path.abspath(logdir), format_strs)

    model = algorithm(policy, env, n_steps=args.steps, verbose=1)
    cb = CheckpointCallback(args.steps, logdir, verbose=1)
    model.learn(total_timesteps=totalsteps, callback=cb)