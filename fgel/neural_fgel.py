import torch

from fgel.baselines.least_squares import OrdinaryLeastSquares
from fgel.generalized_el import GeneralizedEL
from fgel.utils.torch_utils import ModularMLPModel


class NeuralFGEL(GeneralizedEL):
    def __init__(self, model, l2_lambda=1e-6, batch_size=200, dual_func_network_kwargs=None, **kwargs):
        super().__init__(model=model, theta_optim='oadam_gda', **kwargs)

        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.dual_func_network_kwargs = self._update_default_dual_func_network_kwargs(dual_func_network_kwargs)

        self.batch_training = True
        self.optimize_step = self._gradient_descent_ascent_step

    def _init_dual_func(self):
        self.dual_func = ModularMLPModel(**self.dual_func_network_kwargs)

    def _update_default_dual_func_network_kwargs(self, dual_func_network_kwargs):
        dual_func_network_kwargs_default = {
            "input_dim": self.dim_z,
            "layer_widths": [50, 20],
            "activation": torch.nn.LeakyReLU,
            "num_out": self.model.psi_dim,
        }
        if dual_func_network_kwargs is not None:
            dual_func_network_kwargs_default.update(dual_func_network_kwargs)
        return dual_func_network_kwargs_default

    def objective(self, x, z, *args):
        hz = self.dual_func(z)
        h_psi = torch.einsum('ik, ik -> i', hz, self.model.psi(x))
        moment = torch.mean(self.gel_function(h_psi))
        if self.l2_lambda > 0:
            l_reg = self.l2_lambda * torch.mean(hz ** 2)
        else:
            l_reg = 0
        return moment, -moment + l_reg


if __name__ == '__main__':
    from experiments.exp_heteroskedastic import run_heteroskedastic_n_times

    results = run_heteroskedastic_n_times(theta=1.7, noise=1.0, n_train=200, repititions=5,
                                          estimatortype=OrdinaryLeastSquares)
    print('Pretrained Thetas: ', results['theta'])

    estimatorkwargs = dict(max_num_epochs=50000, eval_freq=2000, divergence='log')
    results = run_heteroskedastic_n_times(theta=1.7, noise=1.0, n_train=2000, repititions=2,
                                          estimatortype=NeuralFGEL, estimatorkwargs=estimatorkwargs)
    print('Thetas: ', results['theta'])
