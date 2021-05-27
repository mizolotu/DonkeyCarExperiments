import numpy as np
import os, pandas
import os.path as osp
import argparse as arp
import plotly.io as pio
import plotly.graph_objs as go

from reinforcement_learning.gym.envs.donkey_car.donkey_env import DonkeyEnv
from plot_results import moving_average, generate_line_scatter
from off_policy_experiments import algorithm_list as alg_list

schema_list = {
    0: {
        'alg': 0,
        'env': 0,
        'xlimit': 1000000,
        'dirs': ['MlpPolicy_expert'],
        'names': ['No pretraining']
    },
    1: {
        'alg': 0,
        'env': 1,
        'xlimit': 1000000,
        'dirs': ['MlpPolicy_ac'],
        'names': ['Offline learning']
    },
    2: {
        'alg': 1,
        'env': 0,
        'xlimit': 1000000,
        'dirs': ['MlpPolicy_expert'],
        'names': ['No pretraining']
    },
    3: {
        'alg': 1,
        'env': 1,
        'xlimit': 1000000,
        'dirs': ['MlpPolicy_expert'],
        'names': ['No pretraining']
    }
}

if __name__ == '__main__':

    # params

    parser = arp.ArgumentParser(description='Plot progress')
    parser.add_argument('-s', '--schema', help='Schema', default='1,3')
    parser.add_argument('-i', '--input', help='Input', default='models')
    parser.add_argument('-o', '--output', help='Output', default='figures')
    args = parser.parse_args()

    colors = ['rgb(64,120,211)', 'rgb(0,100,80)', 'rgb(237,2,11)', 'rgb(255,165,0)', 'rgb(139,0,139)', 'rgb(0,51,102)']

    if args.schema is not None:
        schema_inds = args.schema.split(',')
    else:
        schema_inds = [i for i in range(len(schema_list))]

    for s_idx in schema_inds:
        print(s_idx)
        schema = schema_list[int(s_idx)]
        env_name = f"DonkeyEnv_{schema['env']}"
        alg = alg_list[schema['alg']]
        xlimit = schema['xlimit']
        dirs = schema['dirs']
        names = schema['names']
        data = []

        for dir in dirs:
            print(dir)
            fname = osp.join(args.input, env_name, alg.__name__, dir, 'progress.csv')
            p = pandas.read_csv(fname, delimiter=',', dtype=float)
            y = p['ep_reward_mean'].values
            if 'total_timesteps' in p.keys():
                x = p['total_timesteps'].values
            else:
                x = p['total timesteps'].values
            y = moving_average(y.reshape(len(y), 1)).reshape(x.shape)
            data.append([x, y])
        traces, layout = generate_line_scatter(names, data, colors, xlabel='Time steps', ylabel='Reward', show_legend=True, xrange=[0, xlimit])

        # save results

        ftypes = ['png', 'pdf']
        if not osp.exists(args.output):
            os.mkdir(args.output)
        env_figs = osp.join(args.output, env_name)
        if not osp.exists(env_figs):
            os.mkdir(env_figs)
        fig_fname = osp.join(env_figs, f'{alg.__name__}_{s_idx}')
        fig = go.Figure(data=traces, layout=layout)
        for ftype in ftypes:
            pio.write_image(fig, '{0}.{1}'.format(fig_fname, ftype))