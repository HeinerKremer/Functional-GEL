import argparse
import copy
import json
import os

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from fgel.estimation import estimation


def run_experiment(experiment, exp_params, n_train, estimation_method, estimator_kwargs=None,
                   hyperparams=None, seed0=12345):
    """
    Runs experiment with specified estimator and choice of hyperparams and returns the best model and the
    corresponding error measures.
    """
    np.random.seed(seed0)
    torch.random.manual_seed(seed0+1)

    exp = experiment(**exp_params)
    exp.prepare_dataset(n_train=n_train, n_val=n_train, n_test=20000)
    model = exp.init_model()

    trained_model, full_results = estimation(model=model,
                                             train_data=exp.train_data,
                                             moment_function=exp.moment_function,
                                             estimation_method=estimation_method,
                                             estimator_kwargs=estimator_kwargs, hyperparams=hyperparams,
                                             validation_data=exp.val_data, val_loss_func=exp.validation_loss,
                                             verbose=True)

    test_risks = []
    parameter_mses = []

    # Evaluate test metrics for all models (independent of hyperparam search)
    for model in full_results['models']:
        test_risks.append(float(exp.eval_risk(model, exp.test_data)))
        if exp.get_true_parameters is not None:
            parameter_mses.append(float(np.mean(np.square(np.squeeze(model.get_parameters()) - np.squeeze(exp.get_true_parameters())))))
        else:
            parameter_mses.append(0)

    # Models can't be saved as json and are not needed anymore
    del full_results['models']

    result = {'test_risk_optim': test_risks[full_results['best_index']],
              'parameter_mse_optim': parameter_mses[full_results['best_index']],
              'test_risks': test_risks, 'parameter_mses': parameter_mses,
              'full_results': full_results,}
    return result


def run_experiment_repeated(experiment, exp_params, n_train, estimation_method, estimator_kwargs=None, hyperparams=None,
                            repititions=2, seed0=12345, parallel=True, filename=None):
    """
    Runs the same experiment `repititions` times and computes statistics.
    """
    if parallel:
        results = run_parallel(experiment=experiment, exp_params=exp_params, n_train=n_train,
                               estimation_method=estimation_method, estimator_kwargs=estimator_kwargs,
                               hyperparams=hyperparams, repititions=repititions, seed0=seed0)
        results = list(results)
    else:
        print('Using sequential debugging mode.')
        results = []
        for i in range(repititions):
            stats = run_experiment(experiment=experiment, exp_params=exp_params, n_train=n_train,
                                   estimation_method=estimation_method, estimator_kwargs=estimator_kwargs,
                                   hyperparams=hyperparams, seed0=seed0+i)
            results.append(stats)

    if filename is not None:
        if estimation_method.split('-')[-1] in {'chi2', 'kl', 'log'} or 'divergence' in {hyperparams}:
            divergence = f'-{hyperparams["divergence"]}'
        else:
            divergence = ""
        prefix = f"results/{str(experiment.__name__)}/{str(experiment.__name__)}_method={estimation_method}{divergence}_n={n_train}"
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        print(filename)
        print(prefix + str(filename) + ".json")
        with open(prefix + filename + ".json", "w") as fp:
            json.dump(results, fp)
    return results


def run_parallel(experiment, exp_params, n_train, estimation_method, estimator_kwargs, hyperparams, repititions, seed0):
    experiment_list = [copy.deepcopy(experiment) for _ in range(repititions)]
    exp_params_list = [copy.deepcopy(exp_params) for _ in range(repititions)]
    n_train_list = [copy.deepcopy(n_train) for _ in range(repititions)]
    estimator_method_list = [copy.deepcopy(estimation_method) for _ in range(repititions)]
    estimator_kwargs_list = [copy.deepcopy(estimator_kwargs) for _ in range(repititions)]
    hyperparams_list = [copy.deepcopy(hyperparams) for _ in range(repititions)]
    seeds = [seed0+i for i in range(repititions)]

    with ProcessPoolExecutor(min(multiprocessing.cpu_count(), repititions)) as ex:
        results = ex.map(run_experiment, experiment_list, exp_params_list, n_train_list, estimator_method_list,
                         estimator_kwargs_list, hyperparams_list, seeds)
    return results


if __name__ == "__main__":
    from fgel.default_config import experiments

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_sequential', action='store_true')
    parser.add_argument('--experiment', type=str, default='heteroskedastic')
    parser.add_argument('--exp_option', default=None)
    parser.add_argument('--n_train', type=int, default=100)
    parser.add_argument('--method', type=str, default='KernelVMM')
    parser.add_argument('--rollouts', type=int, default=2)

    args = parser.parse_args()

    exp_info = experiments[args.experiment]

    if args.exp_option is not None:
        exp_info['exp_params'] = {list(exp_info['exp_params'].keys())[0]: args.exp_option}
        filename = '_' + args.exp_option
    else:
        filename = ''

    results = run_experiment_repeated(experiment=exp_info['exp_class'],
                                      exp_params=exp_info['exp_params'],
                                      n_train=args.n_train,
                                      estimation_method=args.method,
                                      repititions=args.rollouts,
                                      parallel=not args.run_sequential,
                                      filename=filename)
    print(results)
