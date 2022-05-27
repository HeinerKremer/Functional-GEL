import math
import time
import torch
from fgel.generalized_el import GeneralizedEL
from fgel.utils.oadam import OAdam
from fgel.utils.torch_utils import BatchIter, ModularMLPModel

import matplotlib.pyplot as plt


class NeuralFGEL(GeneralizedEL):
    def __init__(self, model, l2_lambda=1e-6, batch_size=200, dual_network_kwargs=None, dual_optim_args=None, **kwargs):
        super().__init__(model=model, **kwargs)

        if dual_optim_args is None:
            dual_optim_args = {"lr": 5 * 5e-4}

        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.dual_optim_args = dual_optim_args

        self.dual_network_kwargs = self._update_default_dual_network_kwargs(dual_network_kwargs)
        self.dual = ModularMLPModel(**self.dual_network_kwargs)
        self.dual_optimizer = None

        self.batch_training = True
        self.optimize_step = self._gradient_descent_ascent_step

    def _set_optimizers(self, param_container=None):
        self.dual_optimizer = OAdam(params=self.dual.parameters(), lr=self.dual_optim_args["lr"], betas=(0.5, 0.9))
        self.theta_optimizer = OAdam(params=self.model.parameters(), lr=self.theta_optim_args["lr"], betas=(0.5, 0.9))

    def _update_default_dual_network_kwargs(self, dual_network_kwargs):
        dual_network_kwargs_default = {
            "input_dim": self.dim_z,
            "layer_widths": [50, 20],
            "activation": torch.nn.LeakyReLU,
            "num_out": self.model.psi_dim,
        }
        if dual_network_kwargs is not None:
            dual_network_kwargs_default.update(dual_network_kwargs)
        return dual_network_kwargs_default

    def _init_training(self, x_tensor, z_tensor, z_val_tensor=None):
        self._set_kernel(z_tensor, z_val_tensor)
        self._set_optimizers()
        if self.pretrain:
            self._pretrain_theta(x=x_tensor, z=z_tensor)

    def objective(self, x, z, *args):
        hz = self.dual(z)
        h_psi = torch.einsum('ik, ik -> i', hz, self.model.psi(x))
        moment = torch.mean(self.gel_function(h_psi))
        if self.l2_lambda > 0:
            l_reg = self.l2_lambda * torch.mean(hz ** 2)
        else:
            l_reg = 0
        return moment, -moment + l_reg

    def _gradient_descent_ascent_step(self, x_tensor, z_tensor):
        theta_obj, dual_obj = self.objective(x_tensor, z_tensor)

        # update theta
        self.theta_optimizer.zero_grad()
        theta_obj.backward(retain_graph=True)
        self.theta_optimizer.step()

        # update f network
        self.dual_optimizer.zero_grad()
        dual_obj.backward()
        self.dual_optimizer.step()
        return float(theta_obj.detach().numpy())


if __name__ == '__main__':
    from experiments.exp_heteroskedastic import run_heteroskedastic_n_times

    estimatorkwargs = dict(max_num_epochs=5000, eval_freq=500, divergence='chi2')
    results = run_heteroskedastic_n_times(theta=1.7, noise=1.0, n_train=200, repititions=5,
                                         estimatortype=NeuralFGEL, estimatorkwargs=estimatorkwargs)
    print('Thetas: ', results['theta'])
