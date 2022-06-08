from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment
from experiments.exp_network_iv import NetworkIVExperiment

from fgel.baselines.least_squares import OrdinaryLeastSquares
from fgel.baselines.kernel_mmr import KernelMMR
from fgel.baselines.kernel_vmm import KernelVMM
from fgel.baselines.neural_vmm import NeuralVMM
from fgel.baselines.sieve_minimum_distance import SMDIdentity, SMDHeteroskedastic
from fgel.kernel_fgel import KernelFGEL
from fgel.neural_fgel import NeuralFGEL


experiments = {
    'heteroskedastic':
        {
            'exp_class': HeteroskedasticNoiseExperiment,
            'exp_params': {'theta': [1.7],
                           'noise': 1.0,
                           'heteroskedastic': True, },
            'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
        },

    'network_iv':
        {
            'exp_class': NetworkIVExperiment,
            'exp_params': {'ftype': None},
            'n_train': [200],
        },
}

methods = {
#     'OrdinaryLeastSquares':
#         {
#             'estimator_class': OrdinaryLeastSquares,
#             'estimator_kwargs': {},
#             'hyperparams': {},
#         },
#
#     'KernelMMR':
#         {
#             'estimator_class': KernelMMR,
#             'estimator_kwargs': {},
#             'hyperparams': {},
#         },
#
#     'SieveMinimumDistance':
#         {
#             'estimator_class': SMDHeteroskedastic,
#             'estimator_kwargs': {},
#             'hyperparams': {}
#         },
#
#     'KernelVMM':
#         {
#             'estimator_class': KernelVMM,
#             'estimator_kwargs': {},
#             'hyperparams': {'alpha': [1e-8, 1e-6, 1e-4]}
#         },
#
#     'NeuralVMM':
#         {
#             'estimator_class': NeuralVMM,
#             'estimator_kwargs': {"batch_size": 200,
#                                  "max_num_epochs": 20000,
#                                  "burn_in_cycles": 5,
#                                  "eval_freq": 100,
#                                  "max_no_improve": 3,
#                                  },
#             'hyperparams': {"l2_lambda": [0, 1e-4, 1e-2, 1e0]}
#         },
}

for divergence in ['chi2', 'kl', 'log']:
    methods[f'KernelFGEL-{divergence}'] = {
        'estimator_class': KernelFGEL,
        'estimator_kwargs': {
                "divergence": divergence,
                "dual_optim": 'lbfgs',
                "theta_optim": 'lbfgs',
                "eval_freq": 100,
                "max_num_epochs": 20000,},
        'hyperparams': {'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]}
        }

for divergence in ['chi2', 'kl', 'log']:
    methods[f'KernelFGEL-{divergence}-oadam'] = {
        'estimator_class': KernelFGEL,
        'estimator_kwargs': {
            "divergence": divergence,
            'theta_optim': 'oadam_gda',
            "max_num_epochs": 20000,
            "burn_in_cycles": 5,
            "eval_freq": 100,
            "max_no_improve": 3, },
        'hyperparams': {'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]}
        }

# for divergence in ['chi2', 'kl', 'log']:
#     methods[f'NeuralFGEL-{divergence}'] = {
#         'estimator_class': NeuralFGEL,
#         'estimator_kwargs': {
#             "divergence": divergence,
#             "batch_size": 200,
#             "max_num_epochs": 20000,
#             "burn_in_cycles": 5,
#             "eval_freq": 100,
#             "max_no_improve": 3,},
#         'hyperparams': {"l2_lambda": [0, 1e-4, 1e-2, 1e0]}
#         }
