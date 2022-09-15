import torch

from fgel.baselines.least_squares import OrdinaryLeastSquares
from fgel.generalized_el import GeneralizedEL
from fgel.utils.torch_utils import ModularMLPModel, Parameter


class NeuralFGEL(GeneralizedEL):
    def __init__(self, model, reg_param=1e-6, batch_size=200, dual_func_network_kwargs=None, **kwargs):
        super().__init__(model=model, theta_optim='oadam_gda', **kwargs)

        self.batch_size = batch_size
        self.l2_lambda = reg_param
        self.dual_func_network_kwargs = self._update_default_dual_func_network_kwargs(dual_func_network_kwargs)

        self.batch_training = True
        self.optimize_step = self._gradient_descent_ascent_step

    def _init_dual_params(self):
        self.dual_moment_func = ModularMLPModel(**self.dual_func_network_kwargs)
        self.all_dual_params = list(self.dual_moment_func.parameters())

    def _update_default_dual_func_network_kwargs(self, dual_func_network_kwargs):
        dual_func_network_kwargs_default = {
            "input_dim": self.dim_z,
            "layer_widths": [50, 20],
            "activation": torch.nn.LeakyReLU,
            "num_out": self.model.dim_psi,
        }
        if dual_func_network_kwargs is not None:
            dual_func_network_kwargs_default.update(dual_func_network_kwargs)
        return dual_func_network_kwargs_default

    def objective(self, x, z, *args):
        hz = self.dual_moment_func(z)
        h_psi = torch.einsum('ik, ik -> i', hz, self.model.psi(x))
        moment = torch.mean(self.gel_function(h_psi))
        if self.l2_lambda > 0:
            l_reg = self.l2_lambda * torch.mean(hz ** 2)
        else:
            l_reg = 0
        return moment, -moment + l_reg

    # def objective(self, x, z, *args):
    #     hz = self.dual_func(z)
    #     h_psi = torch.einsum('ik, ik -> i', hz, self.model.psi(x))
    #     moment = torch.mean(self.gel_function(h_psi + self.dual_normalization.params))
    #     if self.l2_lambda > 0:
    #         l_reg = self.l2_lambda * torch.mean(hz ** 2)
    #     else:
    #         l_reg = 0
    #     return moment, -moment + l_reg - self.dual_normalization.params


if __name__ == '__main__':
    from experiments.tests import test_cmr_estimator
    test_cmr_estimator(estimation_method='NeuralFGEL', n_runs=2, hyperparams={'divergence': ['chi2']})
    test_cmr_estimator(estimation_method='NeuralFGEL', n_runs=2, hyperparams={'divergence': ['kl']})
    test_cmr_estimator(estimation_method='NeuralFGEL', n_runs=2, hyperparams={'divergence': ['log']})
