import numpy as np
import os, pandas
import os.path as osp
import argparse as arp
import plotly.io as pio
import plotly.graph_objs as go

from on_policy_experiments import env_list
from on_policy_experiments import algorithm_list as on_policy_algs
from off_policy_experiments import algorithm_list as off_policy_algs

alg_list = on_policy_algs + off_policy_algs

def moving_average(x, step=1, window=10):

    seq = []
    n = x.shape[0]

    for i in np.arange(0, n, step):
        idx = np.arange(np.maximum(0, i - window), np.minimum(n - 1, i + window + 1))
        seq.append(np.mean(x[idx, :], axis=0))

    return np.vstack(seq)

def generate_line_scatter(names, values, colors, xlabel, ylabel, xrange, show_legend=True):

    traces = []

    for i in range(len(names)):
        x = values[i][0].tolist()
        y = values[i][1].tolist()

        traces.append(
            go.Scatter(
                x=x,
                y=y,
                line=dict(color=colors[i]),
                mode='lines',
                showlegend=show_legend,
                name=names[i],
            )
        )

    layout = go.Layout(
        template='plotly_white',
        xaxis=dict(
            title=xlabel,
            showgrid=True,
            showline=False,
            showticklabels=True,
            ticks='outside',
            zeroline=False,
            range=xrange
        ),
        yaxis=dict(
            title=ylabel,
            showgrid=True,
            showline=False,
            showticklabels=True,
            ticks='outside',
            zeroline=False
        ),
    )

    return traces, layout

schema_list = {
    0: {
        'alg': 0,
        'env': 0,
        'xlimit': 6000000,
        'dirs': ['policy_0_pure', 'policy_1_pure', 'policy_2_pure'],
        'names': ['0/2 shared layers', '1/2 shared layers', '2/2 shared layers']
    },
    1: {
        'alg': 0,
        'env': 1,
        'xlimit': 4000000,
        'dirs': ['policy_0_pure', 'policy_1_pure', 'policy_2_pure'],
        'names': ['0/2 shared layers', '1/2 shared layers', '2/2 shared layers']
    },
    2: {
        'alg': 0,
        'env': 2,
        'xlimit': 8000000,
        'dirs': ['policy_0_pure', 'policy_1_pure', 'policy_2_pure'],
        'names': ['0/2 shared layers', '1/2 shared layers', '2/2 shared layers']
    },
    3: {
        'alg': 0,
        'env': 3,
        'xlimit': 10000000,
        'dirs': ['policy_0_pure', 'policy_1_pure', 'policy_2_pure'],
        'names': ['0/2 shared layers', '1/2 shared layers', '2/2 shared layers']
    },
    4: {
        'alg': 1,
        'env': 0,
        'xlimit': 1000000,
        'dirs': ['policy_0_pure', 'policy_1_pure', 'policy_2_pure'],
        'names': ['0/2 shared layers', '1/2 shared layers', '2/2 shared layers']
    },
    5: {
        'alg': 1,
        'env': 1,
        'xlimit': 1000000,
        'dirs': ['policy_0_pure', 'policy_1_pure', 'policy_2_pure'],
        'names': ['0/2 shared layers', '1/2 shared layers', '2/2 shared layers']
    },
    6: {
        'alg': 1,
        'env': 2,
        'xlimit': 2000000,
        'dirs': ['policy_0_pure', 'policy_1_pure', 'policy_2_pure'],
        'names': ['0/2 shared layers', '1/2 shared layers', '2/2 shared layers']
    },
    7: {
        'alg': 1,
        'env': 3,
        'xlimit': 10000000,
        'dirs': ['policy_0_pure', 'policy_1_pure', 'policy_2_pure'],
        'names': ['0/2 shared layers', '1/2 shared layers', '2/2 shared layers']
    },
}

if __name__ == '__main__':

    # params

    parser = arp.ArgumentParser(description='Plot progress')
    parser.add_argument('-s', '--schema', help='Schema', default='0,1,2,4,5,6')
    parser.add_argument('-i', '--input', help='Input', default='models')
    parser.add_argument('-o', '--output', help='Output', default='figures')
    args = parser.parse_args()

    colors = ['rgb(64,120,211)', 'rgb(0,100,80)', 'rgb(237,2,11)', 'rgb(255,165,0)', 'rgb(139,0,139)', 'rgb(0,51,102)']

    if args.schema is not None:
        schema_inds = args.schema.split(',')
    else:
        schema_inds = [i for i in range(len(schema_list))]

    for s_idx in schema_inds:
        schema = schema_list[int(s_idx)]
        env = env_list[schema['env']]
        alg = alg_list[schema['alg']]
        xlimit = schema['xlimit']
        dirs = schema['dirs']
        names = schema['names']
        data = []

        for dir in dirs:
            fname = osp.join(args.input, env.__name__, alg.__name__, dir, 'progress.csv')
            p = pandas.read_csv(fname, delimiter=',', dtype=float)
            y = p['ep_reward_mean'].values
            x = p['total_timesteps'].values
            y = moving_average(y.reshape(len(y), 1)).reshape(x.shape)
            data.append([x, y])
        traces, layout = generate_line_scatter(names, data, colors, xlabel='Time steps', ylabel='Reward', show_legend=True, xrange=[0, xlimit])

        # save results

        ftypes = ['png', 'pdf']
        if not osp.exists(args.output):
            os.mkdir(args.output)
        env_figs = osp.join(args.output, env.__name__)
        if not osp.exists(env_figs):
            os.mkdir(env_figs)
        fig_fname = osp.join(env_figs, f'{alg.__name__}_{s_idx}')
        fig = go.Figure(data=traces, layout=layout)
        for ftype in ftypes:
            pio.write_image(fig, '{0}.{1}'.format(fig_fname, ftype))