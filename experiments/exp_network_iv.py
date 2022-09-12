from functools import partial

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from experiments.abstract_experiment import AbstractExperiment
from fgel.baselines.kernel_mmr import KernelMMR
from fgel.baselines.least_squares import OrdinaryLeastSquares
from fgel.baselines.neural_vmm import NeuralVMM
from fgel.kernel_fgel import KernelFGEL
from fgel.neural_fgel import NeuralFGEL

z_dim = 2


class NetworkModel(nn.Module):
    """A multilayer perceptron to approximate functions in the IV problem"""

    def __init__(self):
        nn.Module.__init__(self)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 20),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(20, 3),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(3, 1)
        )
        self.psi_dim = 1
        self.dim_z = z_dim

    def forward(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        return self.model(t)

    def psi(self, x):
        t, y = torch.Tensor(x[0]), torch.Tensor(x[1])
        return self.forward(t) - y

    def initialize(self):
        pass

    def is_finite(self):
        isnan = np.sum(np.isnan(self.get_parameters()))
        isfinite = np.sum(np.isfinite(self.get_parameters()))
        return (not isnan) and isfinite

    def get_parameters(self):
        return 0


class NetworkIVExperiment(AbstractExperiment):
    def __init__(self, ftype='sin'):
        super().__init__(self, theta_dim=None, z_dim=z_dim)
        self.ftype = ftype
        self.func = self.set_function()
        self.z_dim = z_dim

    def init_model(self):
        return NetworkModel()

    def get_true_parameters(self):
        return 0

    def generate_data(self, n_sample, split=None):
        """Generates train, validation and test data"""
        e = np.random.normal(loc=0, scale=1.0, size=[n_sample, 1])
        gamma = np.random.normal(loc=0, scale=0.1, size=[n_sample, 1])
        delta = np.random.normal(loc=0, scale=0.1, size=[n_sample, 1])

        z = np.random.uniform(low=-3, high=3, size=[n_sample, self.z_dim])
        t = np.reshape(z[:, 0], [-1, 1]) + e + gamma
        y = self.func(t) + e + delta
        x = [t, y]
        return x, z

    def eval_test_risk(self, model, x_test=None, t_test=None):
        if t_test is None:
            t_test = x_test[0]
        g_test = self.func(t_test)
        g_test_pred = model.forward(t_test).detach().cpu().numpy()
        mse = float(((g_test - g_test_pred) ** 2).mean())
        return mse

    def set_function(self):
        if self.ftype == 'linear':
            def func(x):
                return x
        elif self.ftype == 'sin':
            def func(x):
                return np.sin(x)
        elif self.ftype == 'step':
            def func(x):
                return np.asarray(x > 0, dtype=float)
        elif self.ftype == 'abs':
            def func(x):
                return np.abs(x)
        else:
            raise NotImplementedError
        return func

    def show_function(self, model=None, x_test=None, x_train=None, title=None):
        mse = self.eval_test_risk(model, x_test=x_test)
        t = x_test[0]

        g_true = self.func(t)
        g_test_pred = model.forward(t).detach().cpu().numpy()

        order = np.argsort(t[:, 0])
        fig, ax = plt.subplots(1)
        ax.plot(t[order], g_true[order], label='True function', color='y')
        if x_train is not None:
            ax.scatter(x_train[0], x_train[1], label='Data', s=6)

        if model is not None:
            ax.plot(t[order], g_test_pred[order], label='Model prediction', color='r')
        ax.legend()
        ax.set_title(f'mse={mse:.1e}')
        plt.show()


if __name__ == '__main__':
    exp = NetworkIVExperiment(ftype='abs')
    exp.prepare_dataset(n_train=2000, n_val=2000, n_test=20000)
    model = exp.init_model()


    # estimator = NeuralFGEL(model=model, divergence='chi2', max_num_epochs=5000,)
    estimator = OrdinaryLeastSquares(model=model)
    estimator.train(exp.x_train, exp.z_train, exp.x_val, exp.z_val, debugging=True)
    exp.show_function(model, exp.x_test)

