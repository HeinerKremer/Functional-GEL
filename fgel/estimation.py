import copy

import numpy as np
import torch
import torch.nn as nn

from fgel.default_config import methods

mr_estimators = ['OLS', 'KernelEL']
cmr_estimators = ['MMR', 'SMD', 'KernelVMM', 'NeuralVMM', 'KernelFGEL', 'NeuralFGEL']


def estimation(model, train_data, moment_function, estimation_method,
               estimator_kwargs=None, hyperparams=None,
               validation_data=None, val_loss_func=None,
               verbose=True):
    if train_data['z'] is None:
        conditional_mr = False
        print("No instrument data, `train_data['z']`, found. Using unconditional moment restrictions.")
    else:
        conditional_mr = True

    if estimation_method not in (mr_estimators + cmr_estimators):
        raise NotImplementedError(f'Invalid estimation method specified, pick `estimation_method` from '
                                  f'{set(mr_estimators+cmr_estimators)}.')

    if conditional_mr and estimation_method in mr_estimators:
        print("Solving conditional MR problem with method for unconditional MR, "
              "ignoring instrument data `train_data['z']`.")

    if not conditional_mr and estimation_method in cmr_estimators:
        raise RuntimeError("Specified method requires conditional MR but the provided problem is an unconditional MR. "
                           f"Provide instrument data `train_data['z']` or choose `estimation_method` for "
                           f"unconditional MR from {mr_estimators}.")

    # Load estimator and update default estimator kwargs
    method = methods[estimation_method]
    estimator_class = method['estimator_class']
    estimator_kwargs_default = method['estimator_kwargs']
    hyperparams_default = method['hyperparams']

    if estimator_kwargs is not None:
        estimator_kwargs_default.update(estimator_kwargs)
    estimator_kwargs = estimator_kwargs_default

    if hyperparams is not None:
        hyperparams_default.update(hyperparams)
    hyperparams = hyperparams_default

    # Train estimator for different hyperparams and return best model
    trained_model, train_statistics = optimize_hyperparams(model=model,
                                                           moment_function=moment_function,
                                                           estimator_class=estimator_class,
                                                           estimator_kwargs=estimator_kwargs,
                                                           hyperparams=hyperparams,
                                                           train_data=train_data,
                                                           validation_data=validation_data,
                                                           val_loss_func=val_loss_func,
                                                           verbose=verbose)
    return trained_model, train_statistics


def iterate_argument_combinations(argument_dict):
    """
    Iterates over all possible hyperparam combinations contained in a dict e.g. {p1: [1,2,3], p2:[3,4]}.
    """
    args = list(argument_dict.values())
    pools = [tuple(pool) for pool in args]
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield {key: val for key, val in zip(list(argument_dict.keys()), prod)}


def optimize_hyperparams(model, moment_function, estimator_class, estimator_kwargs, hyperparams,
                         train_data, validation_data=None, val_loss_func=None, verbose=False):
    x_train = [train_data['t'], train_data['y']]
    z_train = train_data['z']

    if validation_data is not None:
        x_val = [validation_data['t'], validation_data['y']]
        z_val = validation_data['z']
    else:
        x_val = x_train
        z_val = z_train

    models = []
    params = []
    validation_loss = []

    for hyper in iterate_argument_combinations(hyperparams):
        model_wrapper = ModelWrapper(model=copy.deepcopy(model),
                                     moment_function=moment_function,
                                     dim_y=x_train[1].shape[1], dim_z=z_train.shape[1])
        if verbose:
            print('Running hyperparams: ', f'{hyper}')
        estimator = estimator_class(model=model_wrapper, **hyper, **estimator_kwargs)
        estimator.train(x_train, z_train, x_val, z_val)

        if val_loss_func is None:
            val_loss = estimator.calc_validation_metric(x_val, z_val)
        else:
            val_loss = val_loss_func(model_wrapper.model, validation_data)
        models.append(model_wrapper.model)
        params.append(hyper)
        validation_loss.append(val_loss.detach().numpy())

    best_val = np.argmin(validation_loss)
    best_params = params[best_val]
    if verbose:
        print('Best hyperparams: ', best_params)
    return models[best_val], {'models': models, 'val_loss': validation_loss, 'params': params, 'best_params': best_params}


def fgel_estimation2(model, train_data, moment_function, version='kernel', divergence=None, reg_param=None,
                       validation_data=None, val_loss_func=None, verbose=True):
    if version == 'kernel':
        method_name = 'KernelFGEL'
    elif version == 'neural':
        method_name = 'NeuralFGEL'
    else:
        raise NotImplementedError('Invalid `version` specified. Use either `kernel` or `neural`.')

    estimator_kwargs = methods[method_name]['estimator_kwargs']
    hyperparams = methods[method_name]['hyperparams']

    if divergence is not None:
        hyperparams.update({'divergence': divergence})

    if reg_param is not None:
        hyperparams.update({'reg'})

    trained_model, train_statistics = estimation(model=model,
                                                 train_data=train_data,
                                                 moment_function=moment_function,
                                                 estimation_method=method_name,
                                                 estimator_kwargs=estimator_kwargs,
                                                 hyperparams=hyperparams,
                                                 validation_data=validation_data,
                                                 val_loss_func=val_loss_func,
                                                 verbose=verbose)
    return trained_model, train_statistics


def fgel_estimation(model, train_data, moment_function, version='kernel', divergence=None, reg_param=None,
                       validation_data=None, val_loss_func=None, verbose=True):
    x_train = [train_data['t'], train_data['y']]
    z_train = train_data['z']

    if validation_data is not None:
        x_val = [validation_data['t'], validation_data['y']]
        z_val = validation_data['z']
    else:
        x_val = x_train
        z_val = z_train

    if version == 'kernel':
        method_name = 'KernelFGEL'
    elif version == 'neural':
        method_name = 'NeuralFGEL'
    else:
        raise NotImplementedError('Invalid version specified. Use either `kernel` or `neural`.')

    if divergence is None:
        divergences = ['chi2', 'kl', 'log']
    else:
        divergences = [divergence]

    models = []
    validation_loss = []
    params = []

    for divergence in divergences:
        method = methods[method_name + f'-{divergence}']
        estimator_class = method['estimator_class']
        estimator_kwargs = method['estimator_kwargs']
        hyperparams = method['hyperparams']
        hyper_param_name = list(hyperparams.keys())[0]

        if reg_param is not None:
            hyperparams = {hyper_param_name: [reg_param]}

        for hval in list(hyperparams.values())[0]:
            hyper = {hyper_param_name: hval}
            model_wrapper = ModelWrapper(model=copy.deepcopy(model),
                                         moment_function=moment_function,
                                         dim_y=x_train[1].shape[1], dim_z=z_train.shape[1])
            if verbose:
                print('Running: ', f'divergence={divergence}, reg_param={hval}')
            estimator = estimator_class(model=model_wrapper, **hyper, **estimator_kwargs)
            estimator.train(x_train, z_train, x_val, z_val)

            if val_loss_func is None:
                val_loss = estimator.calc_validation_metric(x_val, z_val)
            else:
                val_loss = val_loss_func(model_wrapper.model, validation_data)
            models.append(model_wrapper.model)
            params.append({'divergence': divergence, **hyper})
            validation_loss.append(val_loss.detach().numpy())
    best_val = np.argmin(validation_loss)
    best_params = params[best_val]
    if verbose:
        print('Best config: ', best_params)
    return models[best_val], {'models': models, 'val_loss': validation_loss, 'params': params, 'best_params': best_params}


def fgel_iv_estimation(model, train_data, version='kernel', divergence=None, reg_param=None,
                       validation_data=None, val_loss_func=None, verbose=True):

    def moment_function(model_evaluation, y):
        return model_evaluation - y

    return fgel_estimation(model=model, train_data=train_data, moment_function=moment_function,
                           version=version, divergence=divergence, reg_param=reg_param,
                           validation_data=validation_data, val_loss_func=val_loss_func, verbose=verbose)


class ModelWrapper(nn.Module):
    def __init__(self, model, moment_function, dim_y, dim_z):
        nn.Module.__init__(self)
        self.model = model
        self.moment_function = moment_function
        self.psi_dim = dim_y
        self.dim_z = dim_z

    def psi(self, data):
        t, y = torch.Tensor(data[0]), torch.Tensor(data[1])
        return self.moment_function(self.model(t), y)

    def get_parameters(self):
        param_tensor = self.model.parameters().data
        return param_tensor.detach().numpy()

    def is_finite(self):
        isnan = np.sum(np.isnan(self.get_parameters()))
        isfinite = np.sum(np.isfinite(self.get_parameters()))
        return (not isnan) and isfinite