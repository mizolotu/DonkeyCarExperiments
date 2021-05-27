import argparse as arp
import os
import numpy as np
import pandas

from reinforcement_learning import logger
from reinforcement_learning.gym.envs.donkey_car.donkey_env import DonkeyEnv
from reinforcement_learning.common.callbacks import CheckpointCallback

from reinforcement_learning.common.vec_env.subproc_vec_env import SubprocVecEnv
from reinforcement_learning.ddpg.policies import MlpPolicy as ddpg_mlp_policy
from reinforcement_learning.ddpg.policies import CnnPolicy as ddpg_cnn_policy
from reinforcement_learning.sac.policies import MlpPolicy as sac_mlp_policy
from reinforcement_learning.sac.policies import CnnPolicy as sac_cnn_policy

from reinforcement_learning.ddpg.ddpg import DDPG as ddpg
from reinforcement_learning.sac.sac import SAC as sac

from train_expert_donkey import make_env
from on_policy_experiments import generate_traj, find_checkpoint_with_highest_explained_variance, find_checkpoint_with_latest_date

algorithm_list = [
    ddpg,
    sac
]

policy_list = [
    [ddpg_cnn_policy, ddpg_mlp_policy],
    [sac_cnn_policy, sac_mlp_policy]
]

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Test state-of-art RL alghorithms in OpenAI gym')
    parser.add_argument('-e', '--env', help='Environment index', type=int, default=1)
    parser.add_argument('-s', '--steps', help='Number of episode steps', type=int, default=256)
    parser.add_argument('-u', '--updates', help='Number of updates', type=int, default=10000)
    parser.add_argument('-a', '--algorithm', help='RL algorithm index', type=int, default=0)
    parser.add_argument('-o', '--output', help='Output directory', default='models')
    parser.add_argument('-c', '--cuda', help='Use CUDA', default=False, type=bool)
    parser.add_argument('-t', '--trainer', help='Expert model', default='SAC/MlpPolicy_expert')
    parser.add_argument('-l', '--laps', help='Expert laps', default=100000//256)
    parser.add_argument('-p', '--pretrain', help='Full pretrain', default=True, type=bool)
    args = parser.parse_args()

    if not args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    env_class = DonkeyEnv
    conf = {'exe_path': '/home/mizolotu/DonkeyCar/donkey_sim.x86_64', 'port': 9091}
    level = 'generated_track'
    algorithm = algorithm_list[args.algorithm]
    policy = policy_list[args.algorithm][args.env]
    totalsteps = args.steps * args.updates
    env_fns = [make_env(env_class, level, conf, args.env)]
    env = SubprocVecEnv(env_fns)

    if args.trainer is not None:
        if args.pretrain:
            postfix = 'ac'
        else:
            postfix = 'bc'
        trainer_dir = f'{args.output}/{env_class.__name__}_{args.env}/{args.trainer}'
        #checkpoint_file = f'{args.output}/{env_class.__name__}/{args.trainer}/rl_model_{good_checkpoints[args.env]}_steps.zip'
        checkpoint_file = find_checkpoint_with_latest_date(trainer_dir)
        trainer_model = sac.load(checkpoint_file)
        trainer_model.set_env(env)
        print('Expert model has been successfully loaded from {0}'.format(checkpoint_file))

        expert_trajs = f'{trainer_dir}/trajs.csv'
        try:
            p = pandas.read_csv(expert_trajs, header=None, delimiter=',')
            trajs = p.values
        except Exception as e:
            print(e)
            trajs = []
            for i in range(args.laps):
                print(f'Lap {i + 1}/{args.laps}')
                states, actions, next_states, rewards = generate_traj(env, trainer_model, args.steps)
                for se, ae, ne, re in zip(states, actions, next_states, rewards):
                    trajs.append([])
                    for s, a, n, r in zip(se, ae, ne, re):
                        s = s.reshape(1, -1)[0]
                        n = n.reshape(1, -1)[0]
                        trajs[-1].append(np.hstack([s, a, n, r]))
                    trajs[-1] = np.vstack(trajs[-1])
            trajs = np.vstack(trajs)
            pandas.DataFrame(trajs).to_csv(expert_trajs, index=False, header=False)

        del trainer_model
    else:
        postfix = 'pure'

    logdir = f'{args.output}/{env_class.__name__}_{args.env}/{algorithm.__name__}/{policy.__name__}_{postfix}/'
    format_strs = os.getenv('', 'stdout,log,csv').split(',')
    logger.configure(os.path.abspath(logdir), format_strs)

    model = algorithm(policy, env, n_steps=args.steps, verbose=1)
    if postfix == 'bc':
        model.pretrain(trajs, batch_size=args.steps, n_epochs=1, learning_rate=1e-3)
    elif postfix == 'ac':
        model.full_pretrain(trajs, batch_size=args.steps, n_epochs=1)
    print('Pretraining done!')

    cb = CheckpointCallback(args.steps * args.updates, logdir, verbose=1)
    model.learn(total_timesteps=totalsteps, callback=cb)



