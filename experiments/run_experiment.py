import numpy as np
import torch


def run_experiment(experiment, exp_params, n_train, estimator_class, estimator_kwargs=None,
                   hyperparams=None, validation_metric='mmr', seed0=12345):
    """
    Runs experiment with specified estimator and choice of hyperparams and returns the best model and the
    corresponding error measures.
    """
    if estimator_kwargs is None:
        estimator_kwargs = {}
    if hyperparams is None:
        hyperparams = {None: [None]}
    hypervals = list(hyperparams.values())[0]
    hyperparam = list(hyperparams.keys())[0]

    np.random.seed(seed0)
    torch.random.manual_seed(seed0+1)
    exp = experiment(**exp_params)
    exp.setup_data(n_train=n_train, n_val=n_train, n_test=20000)

    train_risks = []
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
        estimator.fit(exp.x_train, exp.z_train, exp.x_val, exp.z_val)

        models.append(model)

        params.append(float(np.squeeze(model.get_parameters())))
        train_risks.append(float(exp.eval_test_risk(model, exp.x_train)))
        test_risks.append(float(exp.eval_test_risk(model, exp.x_test)))
        mses.append(float(np.mean(np.square(np.squeeze(model.get_parameters()) - np.squeeze(exp.get_true_parameters())))))
        val_mmr.append(float(estimator.calc_val_mmr(exp.x_val, exp.z_val).detach().numpy()))
    if validation_metric == 'mmr':
        val_mmr = np.nan_to_num(val_mmr, nan=np.inf)
        i = np.argmin(val_mmr)
    else:
        raise NotImplementedError
    stats = {'hyperparam': hypervals[i],
             'param': params[i],
             'train_risk': train_risks[i],
             'test_risk': test_risks[i],
             'mse': mses[i],
             'val_mmr': val_mmr[i]}
    return models[i], stats


def run_experiment_repeated(experiment, exp_params, n_train, estimator_class, estimator_kwargs, hyperparams, repititions, seed0=12345):
    hypervals = []
    train_risk = []
    test_risk = []
    mse = []
    val_mmr = []

    for i in range(repititions):
        _, stats = run_experiment(experiment=experiment, exp_params=exp_params, n_train=n_train,
                                  estimator_class=estimator_class, estimator_kwargs=estimator_kwargs,
                                  hyperparams=hyperparams, validation_metric='mmr', seed0=seed0+i)

        hypervals.append(stats['hyperparam'])
        train_risk.append(stats['train_risk'])
        test_risk.append(stats['test_risk'])
        mse.append(stats['mse'])
        val_mmr.append(stats['val_mmr'])

    results = {"mean_square_error": np.mean(mse),
                "std_square_error": np.std(mse),
                "max_square_error": np.max(mse),
                "mean_risk": np.mean(test_risk),
                "std_risk": np.std(test_risk),
                "max_risk": np.max(test_risk),
                "mean_mmr_loss": np.mean(val_mmr),
                "std_mmr_loss": np.std(val_mmr),
               "hyperparam_values": hypervals,
               }
    return results


if __name__ == "__main__":
    from fgel.baselines.least_squares import OrdinaryLeastSquares
    from fgel.kernel_fgel import KernelFGEL
    from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment

    exp_params = {'theta': [1.7], 'noise': 1.0}
    estimator_kwargs = dict(max_num_epochs=100, eval_freq=50,
                            divergence='kl', outeropt='lbfgs', inneropt='lbfgs', inneriters=100)
    hyperparams = {'reg_param': [1e-1, 1e-3, 1e-6, 1e-9]}

    results = run_experiment_repeated(experiment=HeteroskedasticNoiseExperiment,
                            exp_params=exp_params,
                            n_train=200,
                            estimator_class=KernelFGEL,
                            estimator_kwargs=estimator_kwargs,
                            hyperparams=hyperparams,
                            repititions=5)

    print(results)
