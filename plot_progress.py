import os, pandas
import os.path as osp
import argparse as arp
import plotly.io as pio
import plotly.graph_objs as go

import numpy as np

def moving_average(x, step=1, window=10):

    seq = []
    n = x.shape[0]

    for i in np.arange(0, n, step):
        idx = np.arange(np.maximum(0, i - window), np.minimum(n - 1, i + window + 1))
        seq.append(np.mean(x[idx, :], axis=0))

    return np.vstack(seq)

def generate_line_scatter(names, values, colors, xlabel, ylabel, show_legend=True):

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

if __name__ == '__main__':

    # params

    parser = arp.ArgumentParser(description='Plot progress')
    parser.add_argument('-i', '--input', help='Input directory', default='results/BipedalWalker/DDPG')
    parser.add_argument('-o', '--output', help='Output directory', default='figures/BipedalWalker/DDPG')
    args = parser.parse_args()

    colors = [['rgb(64,120,211)'], ['rgb(0,100,80)'], ['rgb(237,2,11)'], ['rgb(255,165,0)', 'rgb(139,0,139)', 'rgb(0,51,102)']]

    fname = osp.join(args.input, 'progress.csv')
    p = pandas.read_csv(fname, delimiter=',', dtype=float)
    r = p['ep_reward_mean'].values
    if 'total_timesteps' in p.keys():
        x = p['total_timesteps'].values
    elif 'total/steps' in p.keys():
        x = p['total/steps'].values

    data = [[[x, r]]]
    names = [['Reward']]
    fnames = ['reward']
    ylabels = ['Reward value']

    for d, n, f, y, c in zip(data, names, fnames, ylabels, colors):

        # generate scatter

        traces, layout = generate_line_scatter(n, d, c, 'Time steps', y, show_legend=True)

        # save results

        if not osp.exists(args.output):
            os.mkdir(args.output)
        ftypes = ['png', 'pdf']
        fig_fname = '{0}/{1}'.format(args.output, f)
        fig = go.Figure(data=traces, layout=layout)
        for ftype in ftypes:
            pio.write_image(fig, '{0}.{1}'.format(fig_fname, ftype))