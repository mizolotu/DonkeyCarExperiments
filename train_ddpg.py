import argparse as arp
import os, pathlib
import os.path as osp


from reinforcement_learning.gym.envs.donkey_car.donkey_env import DonkeyEnv
from reinforcement_learning.ddpg.ddpg import DDPG as ddpg
from reinforcement_learning.common.vec_env.subproc_vec_env import SubprocVecEnv
from reinforcement_learning.ddpg.policies import MlpPolicy
from reinforcement_learning import logger
from reinforcement_learning.common.callbacks import CheckpointCallback

def find_checkpoint_with_latest_date(checkpoint_dir, prefix='rl_model_'):
    checkpoint_files = [item for item in os.listdir(checkpoint_dir) if osp.isfile(osp.join(checkpoint_dir, item)) and item.startswith(prefix) and item.endswith('.zip')]
    checkpoint_fpaths = [osp.join(checkpoint_dir, item) for item in checkpoint_files]
    checkpoint_dates = [pathlib.Path(item).stat().st_mtime for item in checkpoint_fpaths]
    idx = sorted(range(len(checkpoint_dates)), key=lambda k: checkpoint_dates[k])
    return checkpoint_fpaths[idx[-1]]

def make_env(env_class, *args):
    fn = lambda: env_class(*args)
    return fn

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Train RL agent.')
    parser.add_argument('-e', '--exe', default='/home/mizolotu/DonkeyCar/donkey_sim.x86_64', help='Simulator exe')
    parser.add_argument('-p', '--port', help='Port', default=9091)
    parser.add_argument('-n', '--nenvs', help='Number of environments', type=int, default=1)
    parser.add_argument('-s', '--nsteps', help='Number of steps', type=int, default=256)
    parser.add_argument('-u', '--nupdates', help='Number of episodes', type=int, default=1000000)
    parser.add_argument('-l', '--level', help='Level', default='generated_track')
    parser.add_argument('-c', '--checkpoint', help='Checkpoint')  # e.g. 'rl_model_384_steps.zip
    args = parser.parse_args()

    if args.nenvs is not None:
        nenvs = args.nenvs

    env_class = DonkeyEnv
    conf = {'exe_path': args.exe, 'port' : args.port}
    algorithm = ddpg
    policy = MlpPolicy
    total_steps = args.nsteps * args.nupdates

    modeldir = '{0}/{1}/{2}'.format('models', env_class.__name__, algorithm.__name__)
    logdir = '{0}/{1}/{2}'.format('results', env_class.__name__, algorithm.__name__)
    format_strs = os.getenv('', 'stdout,log,csv').split(',')
    logger.configure(os.path.abspath(logdir), format_strs)

    # create environments

    env_fns = [make_env(env_class, args.level, conf) for env_idx in range(nenvs)]
    env = SubprocVecEnv(env_fns)

    if args.checkpoint is None:
        chkpt = find_checkpoint_with_latest_date(modeldir)
    else:
        chkpt = args.checkpoint
    print(chkpt)

    try:

        model = algorithm.load('{0}/{1}'.format(modeldir, chkpt))
        model.set_env(env)
        print('Model has been loaded from {0}!'.format(chkpt))
    except Exception as e:
        print(e)
        print('Could not load the model, a new model will be created!')
        model = algorithm(policy, env, verbose=1)
    finally:
        cb = CheckpointCallback(args.nsteps, modeldir, verbose=1)
        model.learn(total_timesteps=total_steps, callback=cb)