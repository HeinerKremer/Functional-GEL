import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate

LINE_WIDTH = 7.0
COL_WIDTH = 3.333

FIG_SIZE_DOUBLE = (COL_WIDTH / 2, COL_WIDTH / 2 * 4/6)
NEURIPS_LINE_WIDTH = 5.5  # Text width: 5.5in (double figure minus spacing 0.2in).
FIG_SIZE_NEURIPS_DOUBLE = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 4/6)
figsize = (LINE_WIDTH*1.4, LINE_WIDTH/2)

labels = {'SMD': 'SMD',
          'KernelFGEL': 'K-FGEL',
          'NeuralFGEL': 'NN-FGEL',
          'KernelFGEL-chi2': 'K-FGEL',
          'NeuralFGEL-chi2': 'NN-FGEL',
          'KernelFGEL-log': 'K-FGEL',
          'NeuralFGEL-log': 'NN-FGEL',
          'KernelFGEL-kl': 'K-FGEL',
          'NeuralFGEL-kl': 'NN-FGEL',
          'KernelMMR': 'MMR',
          'OLS': 'OLS',
          'KernelVMM': 'K-VMM',
          'NeuralVMM': 'NN-VMM',
          'KernelELKernel': 'K-KEL',
          'KernelELNeural': 'NN-KEL'}


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


def load_and_summarize_results(filename):
    with open(filename, "r") as fp:
        results_and_summary = json.load(fp)

    results = results_and_summary['results']

    hypervals = []
    val_loss = []
    test_risk = []
    mse = []

    for stats in results:
        i = stats['best_index']
        hypervals.append(stats['hyperparam'][i])
        val_loss.append(stats['val_loss'][i])
        test_risk.append(stats['test_risk'][i])
        mse.append(stats['mse'][i])

    results_summarized = {
        "mean_square_error": np.mean(mse),
        "std_square_error": np.std(mse),
        "max_square_error": np.max(mse),
        "mean_risk": np.mean(test_risk),
        "std_risk": np.std(test_risk),
        "max_risk": np.max(test_risk),
        "n_runs": len(results),
        "hyperparam_values_list": hypervals,
        "val_loss_list": val_loss,
        "test_risk_list": test_risk,
        "mse_list": mse,
    }
    return results_summarized


def get_result_for_best_divergence(method, n_train, test_metric, experiment=None, func=None):
    if experiment == 'network_iv':
        opt = f'_{func}'
        experiment = 'results/NetworkIVExperiment/NetworkIVExperiment'
    elif experiment == 'heteroskedastic':
        opt = ''
        experiment = 'results/HeteroskedasticNoiseExperiment/HeteroskedasticNoiseExperiment'
    elif experiment == 'poisson':
        experiment = 'results/PoissonExperiment/PoissonExperiment'
        opt = ''
    else:
        raise NotImplementedError

    test_metrics = []
    validation = []
    for divergence in ['chi2', 'log', 'kl']:
        filename = f"{experiment}_method={method}-{divergence}_n={n_train}{opt}.json"
        try:
            res = load_and_summarize_results(filename)
            test_metrics.append(res[test_metric])
            validation.append(res['val_loss'])
        except FileNotFoundError:
            print(f'No such file or directory: {filename}')
    indices = np.nanargmin(np.asarray(test_metrics), axis=0)
    validation_metrics = np.nanmin(np.asarray(validation), axis=0)
    test_metrics = np.asarray(test_metrics)
    test_metrics = np.asarray([test_metrics[index][i] for i, index in enumerate(indices)])
    return test_metrics, validation_metrics


def get_test_metric_over_sample_size(methods, n_samples, experiment, test_metric='mse', remove_failed=False):
    results = {method: {'mean': [], 'std': []} for method in methods}
    n_samples = np.sort(n_samples)
    for n_train in n_samples:
        for method in methods:
            if method in ['NeuralFGEL', 'KernelFGEL']:
                test_metric, val_metric = get_result_for_best_divergence(method, n_train, experiment=experiment, test_metric=test_metric)
            else:
                if experiment == 'heteroskedastic':
                    experiment = 'results/HeteroskedasticNoiseExperiment/HeteroskedasticNoiseExperiment'
                elif experiment == 'poisson':
                    experiment = 'results/PoissonExperiment/PoissonExperiment'
                else:
                    raise NotImplementedError
                filename = f"{experiment}_method={method}_n={n_train}.json"
                res = load_and_summarize_results(filename)
                test_metric, val_metric = res['mse'], res['val_mmr']
            if remove_failed:
                test_metric = remove_failed_runs(test_metric, val_metric)

            results[method]['mean'].append(np.mean(test_metric))
            results[method]['std'].append(np.std(test_metric) / np.sqrt(len(test_metric)))
    return results


def remove_failed_runs(mses, mmrs, proportion=0.9):
    """The baseline KernelVMM fails sometimes, so we have to remove a few runs, to keep the comparison fair
    we simply remove the same proportion of the worst runs from all methods."""
    indeces = np.argsort(mmrs)
    best = np.asarray(mses)[indeces]
    best_mses = best[:int(proportion * len(best))]
    print('Left out MSE: ', best[int(proportion * len(best)):])
    return best_mses


def plot_results_over_sample_size(methods, n_samples, experiment='heteroskedastic', test_metric='mse', logscale=False, remove_failed=False):
    plt.rcParams.update(NEURIPS_RCPARAMS)
    sns.set_theme()

    marker = ['v', 'o', 's', 'd', 'p', '*', 'h']
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:purple']

    # figsize = (LINE_WIDTH, LINE_WIDTH / 2)
    figsize = (LINE_WIDTH/1.3, LINE_WIDTH / 1.8)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = [ax]

    results = get_test_metric_over_sample_size(methods=methods, n_samples=n_samples, experiment=experiment,
                                               test_metric=test_metric, remove_failed=remove_failed)

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
    ax[0].set_ylim(ymax=1.6, ymin=1e-5)

    plt.legend()
    plt.tight_layout()
    plt.savefig('results/HeteroskedasticNoisePlot.pdf', dpi=200)
    plt.show()


def plot_divergence_comparison(n_samples, logscale=False, remove_failed=False):
    plt.rcParams.update(NEURIPS_RCPARAMS)
    sns.set_theme()
    marker = ['v', 'o', 's', 'd', 'p', '*', 'h']
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:purple']

    figsize = (LINE_WIDTH/1.3, LINE_WIDTH / 1.8)
    fig, ax = plt.subplots(2, 1, figsize=figsize)

    labels = [rf'$\chi^2$', 'KL', 'Log']

    for k, version in enumerate(['Kernel', 'Neural']):
        methods = [f'{version}FGEL-chi2', f'{version}FGEL-kl', f'{version}FGEL-log']
        results = {method: {'mean': [], 'std': []} for method in methods}

        n_samples = np.sort(n_samples)
        for n_train in n_samples:
            for method in methods:
                filename = f"results/HeteroskedasticNoiseExperiment/HeteroskedasticNoiseExperiment_method={method}_n={n_train}.json"
                res = load_and_summarize_results(filename)
                mses, mmrs = res['mse'], res['val_mmr']
                if remove_failed:
                    mses = remove_failed_runs(mses, mmrs)

                results[method]['mean'].append(np.mean(mses))
                results[method]['std'].append(np.std(mses) / np.sqrt(len(mses)))

        for i, (method, res) in enumerate(results.items()):
            ax[k].plot(n_samples, res['mean'], label=labels[i], color=colors[i], marker=marker[i], ms=10)
            ax[k].fill_between(n_samples,
                            np.subtract(res['mean'], res['std']),
                            np.add(res['mean'], res['std']),
                            alpha=0.2,
                            color=colors[i])

        ax[1].set_xlabel('sample size')
        ax[k].set_ylabel(r'$||\theta - \theta_0 ||^2$')
        if logscale:
            ax[k].set_xscale('log')
            ax[k].set_yscale('log')
        #ax[0].set_xlim([1e2, 1e4])
        # ax[k].set_ylim([1e-4, 1e0])

        ax[k].legend()
        ax[k].set_title(f'{version}-FGEL')
    plt.tight_layout()
    plt.savefig('results/DivergenceComparison.pdf', dpi=200)
    plt.show()


def generate_table(n_train, test_metric='test_risk', validation_metric='val_mmr', remove_failed=False, optimizer='lbfgs'):
    methods = ['OrdinaryLeastSquares',
               'SMDHeteroskedastic',
               'KernelMMR',
               'KernelVMM',
               'NeuralVMM',
               'KernelFGEL',
               'NeuralFGEL',
               ]
    # methods = ['KernelFGEL-chi2']

    funcs = ['abs', 'step', 'sin', 'linear']

    results = {func: {model: {} for model in methods} for func in funcs}
    for func in funcs:
        for method in methods:
            if method in ['NeuralFGEL', 'KernelFGEL']:
                test, val = get_result_for_best_divergence(method, n_train, test_metric, validation_metric, func, optimizer=optimizer)
            else:
                filename = f"results/NetworkIVExperiment/NetworkIVExperiment_method={method}_n={n_train}_{func}.json"
                res = load_and_summarize_results(filename, validation_metric)
                test, val = res[test_metric], res[validation_metric]
            if remove_failed:
                test = remove_failed_runs(test, val)

            results[func][method]['mean'] = np.mean(test)
            results[func][method]['std'] = np.std(test) / np.sqrt(len(test))

    row1 = [''] + [f"{labels[model]}" for model in methods] # + ['NN-FGEL'] * 3
    # row2 = ['']*5 + []
    table = [row1]
    for func in funcs:
        table.append([f'{func}'] + [r"${:.2f}\pm{:.2f}$".format(results[func][model]["mean"] * 1e1, results[func][model]["std"] * 1e1) for model in
                      methods])
    print(tabulate(table, tablefmt="latex_raw"))


if __name__ == "__main__":
    kernelfgel_optimizer = 'lbfgs'
    remove_failed = True

    plot_results_over_sample_size(['OrdinaryLeastSquares', 'SMDHeteroskedastic', 'KernelMMR', 'KernelVMM', 'NeuralVMM', 'KernelFGEL', 'NeuralFGEL'], # 'OrdinaryLeastSquares', 'KernelMMR', 'KernelVMM', 'KernelFGEL'],
        # methods=['OrdinaryLeastSquares', 'KernelMMR', 'SMDHeteroskedastic', 'KernelFGEL-chi2', 'KernelVMM', 'NeuralFGEL-log', 'NeuralVMM'],
                                  n_samples=[64, 128, 256, 512, 1024, 2048, 4096],   # [50, 100, 200, 500, 1000, 2000],
                                  validation_metric='val_risk',
                                  logscale=True,
                                  remove_failed=remove_failed,
                                  optimizer=kernelfgel_optimizer
                                  )

    plot_divergence_comparison(n_samples=[64, 128, 256, 512, 1024, 2048, 4096],
                               validation_metric='val_risk',
                               logscale=True,
                               remove_failed=remove_failed,
                               optimizer=kernelfgel_optimizer
                               )

    generate_table(n_train=2000,
                   test_metric='test_risk',
                   validation_metric='val_mmr',
                   remove_failed=False,
                   optimizer=None)
