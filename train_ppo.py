import argparse as arp
import os

from reinforcement_learning.gym.envs import DonkeyEnv
from reinforcement_learning.ppo2.ppo2 import PPO2 as ppo
from reinforcement_learning.common.vec_env.subproc_vec_env import SubprocVecEnv
from reinforcement_learning.common.policies import MlpPolicy
from reinforcement_learning import logger
from reinforcement_learning.common.callbacks import CheckpointCallback


def make_env(env_class, *args):
    fn = lambda: env_class(*args)
    return fn

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Train RL agent.')
    parser.add_argument('-e', '--exe', default='/home/mizolotu/DonkeyCar/donkey_sim.x86_64', help='Simulator exe')
    parser.add_argument('-p', '--port', help='Port', default=9091)
    parser.add_argument('-n', '--nenvs', help='Number of environments', type=int, default=4)
    parser.add_argument('-s', '--nsteps', help='Number of steps', type=int, default=256)
    parser.add_argument('-u', '--nupdates', help='Number of episodes', type=int, default=1000000)
    parser.add_argument('-l', '--level', help='Level', default='generated_track')
    parser.add_argument('-c', '--checkpoint', help='Checkpoint')  # e.g. 'rl_model_384_steps.zip
    args = parser.parse_args()

    if args.nenvs is not None:
        nenvs = args.nenvs

    env_class = DonkeyEnv
    conf = {'exe_path': args.exe, 'port' : args.port}
    algorithm = ppo
    policy = MlpPolicy
    total_steps = args.nsteps * args.nupdates

    modeldir = '{0}/{1}/{2}'.format('models', env_class.__name__, algorithm.__name__)
    logdir = '{0}/{1}/{2}'.format('results', env_class.__name__, algorithm.__name__)
    format_strs = os.getenv('', 'stdout,log,csv').split(',')
    logger.configure(os.path.abspath(logdir), format_strs)

    # create environments

    env_fns = [make_env(env_class, args.level, conf) for env_idx in range(nenvs)]
    env = SubprocVecEnv(env_fns)

    try:
        model = algorithm.load('{0}/{1}'.format(modeldir, args.checkpoint))
        model.set_env(env)
        print('Model has been loaded from {0}!'.format(args.checkpoint))
    except Exception as e:
        print('Could not load the model, a new model will be created!')
        model = algorithm(policy, env, n_steps=args.nsteps, verbose=1)
    finally:
        cb = CheckpointCallback(args.nsteps, modeldir, verbose=1)
        model.learn(total_timesteps=total_steps, callback=cb)