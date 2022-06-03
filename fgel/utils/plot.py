import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


LINE_WIDTH = 7.0
COL_WIDTH = 3.333

FIG_SIZE_DOUBLE = (COL_WIDTH / 2, COL_WIDTH / 2 * 4/6)
NEURIPS_LINE_WIDTH = 5.5  # Text width: 5.5in (double figure minus spacing 0.2in).
FIG_SIZE_NEURIPS_DOUBLE = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 4/6)
figsize = (LINE_WIDTH*1.4, LINE_WIDTH/2)

labels = {'SMDIdentity': 'SMD',
          'SMDHeteroskedastic': 'SMD',
          'KernelFGEL-chi2': 'K-FGEL',
          'NeuralFGEL': 'NN-FGEL',
          'KernelMMR': 'MMR',
          'OrdinaryLeastSquares': 'LSQ',
          'KernelVMM': 'K-VMM',
          'NeuralVMM': 'NN-VMM'}


NEURIPS_RCPARAMS = {
    "figure.autolayout": False,         # Makes sure nothing the feature is neat & tight.
    "figure.figsize": FIG_SIZE_NEURIPS_DOUBLE,
    "figure.dpi": 150,                  # Displays figures nicely in notebooks.
    # Axes params
    "axes.linewidth": 0.5,              # Matplotlib's current default is 0.8.
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,

    "hatch.linewidth": 0.3,
    "xtick.major.width": 0.5,
    'xtick.major.pad': 1.0,
    'xtick.major.size': 1.75,
    'xtick.minor.pad': 1.0,
    'xtick.minor.size': 1.0,

    'ytick.major.pad': 1.0,
    'ytick.major.size': 1.75,
    'ytick.minor.pad': 1.0,
    'ytick.minor.size': 1.0,

    "axes.labelpad": 0.5,
    # Grid
    "grid.linewidth": 0.3,
    # Plot params
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    'errorbar.capsize': 3.0,
    # Font
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "Times New Roman", #""serif",             # use serif rather than sans-serif
    "font.serif": "Times New Roman",    # use "Times New Roman" as the standard font
    "font.size": 8.5,
    "axes.titlesize": 8.5,                # LaTeX default is 10pt font.
    "axes.labelsize": 8.5,                # LaTeX default is 10pt font.
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Legend
    "legend.fontsize": 7,        # Make the legend/label fonts a little smaller
    "legend.frameon": True,              # Remove the black frame around the legend
    "legend.handletextpad": 0.3,
    "legend.borderaxespad": 0.2,
    "legend.labelspacing": 0.1,
    "patch.linewidth": 0.5,
    # PDF
    "pgf.texsystem": "xelatex",         # use Xelatex which is TTF font aware
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": [
        r'\usepackage{fontspec}',
        r'\usepackage{unicode-math}',
        r'\setmainfont{Times New Roman}',
    ],
}


def plot_results_over_sample_size(methods, n_samples, quantity='square_error', logscale=False, divergence=None, remove_failed=False):
    plt.rcParams.update(NEURIPS_RCPARAMS)
    sns.set_theme()

    marker = ['v', 'o', 's', 'd', 'p', '*', 'h']
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:purple']

    results = {method: {'mean': [], 'std': []} for method in methods}
    n_samples = np.sort(n_samples)
    for n_train in n_samples:
        for method in methods:
            filename = f"results/HeteroskedasticNoiseExperiment/HeteroskedasticNoiseExperiment_method={method}_n={n_train}.json"
            with open(filename, "r") as fp:
                res = json.load(fp)
            if not remove_failed:
                results[method]['mean'].append(res['mean_'+quantity])
                results[method]['std'].append(res['std_'+quantity] / np.sqrt(res['n_runs']))
            else:
                indeces = np.argsort(res['val_mmr'])
                best = res['mse'][indeces]
                best = best[:int(0.9 * len(best))]
                results[method]['mean'].append(np.mean(best))
                results[method]['std'].append(np.std(best))

    n_plots = 1
    # figsize = (LINE_WIDTH, LINE_WIDTH / 2)
    figsize = (LINE_WIDTH/1.3, LINE_WIDTH / 1.8)

    fig, ax = plt.subplots(1, n_plots, figsize=figsize)
    ax = [ax]

    for i, (method, res) in enumerate(results.items()):
        ax[0].plot(n_samples, res['mean'], label=labels[method], color=colors[i], marker=marker[i], ms=10)
        ax[0].fill_between(n_samples,
                        np.subtract(res['mean'], res['std']),
                        np.add(res['mean'], res['std']),
                        alpha=0.2,
                        color=colors[i])

    ax[0].set_xlabel('sample size')
    ax[0].set_ylabel(r'$||\theta - \theta_0 ||^2$')
    if logscale:
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
    #ax[0].set_xlim([1e2, 1e4])
    ax[0].set_ylim([1e-4, 1e0])

    plt.legend()
    plt.tight_layout()
    plt.savefig('results/HeteroskedasticNoisePlot.pdf', dpi=200)
    plt.show()


if __name__ == "__main__":
    plot_results_over_sample_size(methods=['OrdinaryLeastSquares', 'KernelMMR', 'SMDHeteroskedastic', 'KernelFGEL-chi2', 'KernelVMM'],
                                  n_samples=[64, 128, 256, 512, 1024, 2048],#[50, 100, 200, 500, 1000, 2000],
                                  quantity='square_error',
                                  logscale=True,
                                  remove_failed=True)
