import argparse
import copy
import json
import os

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def run_experiment(experiment, exp_params, n_train, estimator_class, estimator_kwargs=None,
                   hyperparams=None, seed0=12345):
    """
    Runs experiment with specified estimator and choice of hyperparams and returns the best model and the
    corresponding error measures.
    """
    if estimator_kwargs is None:
        estimator_kwargs = {}
    if hyperparams is None or hyperparams == {}:
        hyperparams = {None: [None]}
    hypervals = list(hyperparams.values())[0]
    hyperparam = list(hyperparams.keys())[0]

    np.random.seed(seed0)
    torch.random.manual_seed(seed0+1)
    exp = experiment(**exp_params)
    exp.setup_data(n_train=n_train, n_val=n_train, n_test=20000)

    train_risks = []
    val_risks = []
    test_risks = []
    mses = []
    val_mmr = []
    params = []
    models = []
    for hyperval in hypervals:
        model = exp.init_model()
        if hyperval is None:
            estimator = estimator_class(model=model, **estimator_kwargs)
        else:
            hparam = {hyperparam: hyperval}
            estimator = estimator_class(model=model, **hparam, **estimator_kwargs)
        estimator.train(exp.x_train, exp.z_train, exp.x_val, exp.z_val)

        models.append(model)

        params.append(float(np.squeeze(model.get_parameters())))
        train_risks.append(float(exp.eval_test_risk(model, exp.x_train)))
        val_risks.append(float(exp.eval_test_risk(model, exp.x_val)))
        test_risks.append(float(exp.eval_test_risk(model, exp.x_test)))
        mses.append(float(np.mean(np.square(np.squeeze(model.get_parameters()) - np.squeeze(exp.get_true_parameters())))))
        val_mmr.append(float(estimator._calc_val_mmr(exp.x_val, exp.z_val).detach().numpy()))
    stats = {'hyperparam': hypervals,
             'param': params,
             'train_risk': train_risks,
             'val_risk': val_risks,
             'test_risk': test_risks,
             'mse': mses,
             'val_mmr': val_mmr}
    return stats


def run_experiment_repeated(experiment, exp_params, n_train, estimator_class, estimator_kwargs, hyperparams,
                            repititions, seed0=12345, parallel=True, filename=None):
    """
    Runs the same experiment `repititions` times and computes statistics.
    """
    if parallel:
        results = run_parallel(experiment=experiment, exp_params=exp_params, n_train=n_train, estimator_class=estimator_class,
                               estimator_kwargs=estimator_kwargs, hyperparams=hyperparams, repititions=repititions, seed0=seed0)
        results = list(results)
    else:
        print('Using sequential debugging mode.')
        results = []
        for i in range(repititions):
            stats = run_experiment(experiment=experiment, exp_params=exp_params, n_train=n_train,
                                   estimator_class=estimator_class, estimator_kwargs=estimator_kwargs,
                                   hyperparams=hyperparams, seed0=seed0+i)
            results.append(stats)

    if filename is not None:
        if str(estimator_class.__name__) in {'KernelFGEL', 'NeuralFGEL'}:
            divergence = f'-{estimator_kwargs["divergence"]}'
        else:
            divergence = ""
        if str(estimator_class.__name__) == 'KernelFGEL':
            optim = f'-{estimator_kwargs["theta_optim"]}'
        else:
            optim = ''
        prefix = f"results/{str(experiment.__name__)}/{str(experiment.__name__)}_method={str(estimator_class.__name__)}{divergence}{optim}_n={n_train}"
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        print(filename)
        print(prefix + str(filename) + ".json")
        with open(prefix + filename + ".json", "w") as fp:
            json.dump(results, fp)
    return results


def run_parallel(experiment, exp_params, n_train, estimator_class, estimator_kwargs, hyperparams, repititions, seed0):
    experiment_list = [copy.deepcopy(experiment) for _ in range(repititions)]
    exp_params_list = [copy.deepcopy(exp_params) for _ in range(repititions)]
    n_train_list = [copy.deepcopy(n_train) for _ in range(repititions)]
    estimator_class_list = [copy.deepcopy(estimator_class) for _ in range(repititions)]
    estimator_kwargs_list = [copy.deepcopy(estimator_kwargs) for _ in range(repititions)]
    hyperparams_list = [copy.deepcopy(hyperparams) for _ in range(repititions)]
    # validation_loss_list = ['mmr' for _ in range(repititions)]
    seeds = [seed0+i for i in range(repititions)]

    with ProcessPoolExecutor(min(multiprocessing.cpu_count(), repititions)) as ex:
        results = ex.map(run_experiment, experiment_list, exp_params_list, n_train_list, estimator_class_list,
                         estimator_kwargs_list, hyperparams_list, seeds)
    return results


def run_all(experiment, repititions, method=None, filename=None):
    """
    Runs all methods for all sample sizes `n_train_list` sequentially `repititions` times. This can be used if one has
    only access to a single machine instead of a computer cluster. Might take a long time to finish.
    """
    from experiments.exp_config import methods, experiments

    exp_info = experiments[experiment]

    if method is not None:
        methods = {method: methods[method]}

    for n_train in exp_info['n_train']:
        for method, estimator_info in methods.items():
            print(f'Running {method} with n_train={n_train}.')
            run_experiment_repeated(experiment=exp_info['exp_class'],
                                    exp_params=exp_info['exp_params'],
                                    n_train=n_train,
                                    estimator_class=estimator_info['estimator_class'],
                                    estimator_kwargs=estimator_info['estimator_kwargs'],
                                    hyperparams=estimator_info['hyperparams'],
                                    repititions=repititions,
                                    filename=filename)


if __name__ == "__main__":
    from experiments.exp_config import methods, experiments

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_all', action='store_true')
    parser.add_argument('--run_sequential', action='store_true')
    parser.add_argument('--experiment', type=str, default='heteroskedastic')
    parser.add_argument('--exp_option', default=None)
    parser.add_argument('--n_train', type=int, default=100)
    parser.add_argument('--method', type=str, default='OrdinaryLeastSquares')
    parser.add_argument('--rollouts', type=int, default=2)

    args = parser.parse_args()

    estimator_info = methods[args.method]
    exp_info = experiments[args.experiment]

    if args.exp_option is not None:
        exp_info['exp_params'] = {list(exp_info['exp_params'].keys())[0]: args.exp_option}
        filename = '_' + args.exp_option
    else:
        filename = ''

    if args.run_all:
        run_all(args.experiment, args.rollouts, args.method, filename)
    else:
        results = run_experiment_repeated(experiment=exp_info['exp_class'],
                                          exp_params=exp_info['exp_params'],
                                          n_train=args.n_train,
                                          estimator_class=estimator_info['estimator_class'],
                                          estimator_kwargs=estimator_info['estimator_kwargs'],
                                          hyperparams=estimator_info['hyperparams'],
                                          repititions=args.rollouts,
                                          parallel=not args.run_sequential,
                                          filename=filename)
        print(results)
