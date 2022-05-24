import math
import time

import numpy as np
import torch

from fgel.abstract_estimation_method import AbstractEstimationMethod
from fgel.utils.oadam import OAdam
from fgel.utils.torch_utils import ModularMLPModel, BatchIter


class NeuralVMM(AbstractEstimationMethod):
    def __init__(self, model, l2_lambda=1e-6, kernel_lambda=0, kernel_args=None,
                 batch_size=200, max_num_epochs=5000, eval_freq=500, max_no_improve=5, burn_in_cycles=5,
                 f_network_kwargs=None , f_optimizer_args=None, theta_optimizer_args=None, pretrain=True,
                 verbose=False):

        if f_optimizer_args is None:
            f_optimizer_args = {"lr": 5 * 5e-4}
        if theta_optimizer_args is None:
            theta_optimizer_args = {"lr": 5e-4}

        self.model = model
        self.dim_z = model.dim_z
        self.kernel_lambda = kernel_lambda
        self.l2_lambda = l2_lambda
        self.f_optim_args = f_optimizer_args
        self.theta_optim_args = theta_optimizer_args

        self.f_network_kwargs = self.update_default_f_network_kwargs(f_network_kwargs)
        self.f = ModularMLPModel(**self.f_network_kwargs)

        self.batch_size = batch_size
        self.max_num_epochs = max_num_epochs
        self.eval_freq = eval_freq
        self.max_no_improve = max_no_improve
        self.burn_in_cycles = burn_in_cycles
        self.pretrain = pretrain

        self.verbose = verbose
        AbstractEstimationMethod.__init__(self, model, kernel_args)

    def set_optimizers(self):
        self.f_optimizer = OAdam(params=self.f.parameters(), lr=self.f_optim_args["lr"], betas=(0.5, 0.9))
        self.theta_optimizer = OAdam(params=self.model.parameters(), lr=self.theta_optim_args["lr"], betas=(0.5, 0.9))

    def update_default_f_network_kwargs(self, f_network_kwargs):
        f_network_kwargs_default = {
            "input_dim": self.dim_z,
            "layer_widths": [50, 20],
            "activation": torch.nn.LeakyReLU,
            "num_out": self.model.psi_dim,
        }
        if f_network_kwargs is not None:
            for key, value in f_network_kwargs.items():
                f_network_kwargs_default[key] = value
        return f_network_kwargs_default

    def _game_objective(self, x, z):
        f_of_z = self.f(z)
        m_vector = (self.model.psi(x) * f_of_z).sum(1)
        moment = m_vector.mean()
        ow_reg = 0.25 * (m_vector ** 2).mean()
        if self.kernel_lambda > 0:
            k_reg_list = []
            for i in range(self.psi_dim):
                l_f = self.kernel_z.detach().numpy()
                w = np.linalg.solve(l_f, f_of_z[:, i].detach().cpu().numpy())
                w = self._to_tensor(w)
                k_reg_list.append((w * f_of_z[:, i]).sum())
            k_reg = 2 * self.kernel_lambda * torch.cat(k_reg_list, dim=0).sum()
        else:
            k_reg = 0
        if self.l2_lambda > 0:
            l_reg = self.l2_lambda * (f_of_z ** 2).mean()
        else:
            l_reg = 0
        return moment, -moment + ow_reg + k_reg + l_reg

    def init_training(self, x_tensor, z_tensor, z_val_tensor=None):
        self.set_kernel(z_tensor, z_val_tensor)
        self.set_optimizers()
        if self.pretrain:
            self._pretrain_theta(x=x_tensor, z=z_tensor)

    def _fit_internal(self, x, z, x_val, z_val, debugging=False):
        n = x[0].shape[0]
        batch_iter = BatchIter(n, self.batch_size)
        x_tensor = self._to_tensor(x)
        z_tensor = self._to_tensor(z)
        x_val_tensor = self._to_tensor(x_val)
        z_val_tensor = self._to_tensor(z_val)

        batches_per_epoch = math.ceil(n / self.batch_size)
        eval_freq_epochs = math.ceil(self.eval_freq / batches_per_epoch)

        self.init_training(x_tensor, z_tensor)

        min_val_loss = float("inf")
        time_0 = time.time()
        num_no_improve = 0
        cycle_num = 0

        for epoch_i in range(self.max_num_epochs):
            self.model.train()
            self.f.train()
            for batch_idx in batch_iter:
                # calculate game objectives
                x_batch = [x_tensor[0][batch_idx], x_tensor[1][batch_idx]]
                z_batch = z_tensor[batch_idx]
                rho_obj, f_obj = self._game_objective(x_batch, z_batch)

                # update rho network
                self.theta_optimizer.zero_grad()
                rho_obj.backward(retain_graph=True)
                self.theta_optimizer.step()

                # update f network
                self.f_optimizer.zero_grad()
                f_obj.backward()
                self.f_optimizer.step()

                if epoch_i % eval_freq_epochs == 0:
                    cycle_num += 1
                    val_mmr_loss = self.calc_val_mmr(x_val, z_val)
                    if self.verbose:
                        val_theta_obj, _ = self._game_objective(x_val_tensor, z_val_tensor)
                        print("epoch %d, theta-obj=%f, val-mmr-loss=%f"
                              % (epoch_i, val_theta_obj, val_mmr_loss))
                    if val_mmr_loss < min_val_loss:
                        min_val_loss = val_mmr_loss
                        num_no_improve = 0
                    elif cycle_num > self.burn_in_cycles:
                        num_no_improve += 1
                    if num_no_improve == self.max_no_improve:
                        break
            if self.verbose:
                print("time taken:", time.time() - time_0)


if __name__ == "__main__":
    from experiments.exp_heteroskedastic import run_heteroskedastic_n_times

    estimatorkwargs = dict(dim_z=1, max_num_epochs=5000, eval_freq=50, )
    results = run_heteroskedastic_n_times(theta=1.7, noise=1.0, n_train=200, repititions=10,
                                          estimatortype=NeuralVMM, estimatorkwargs=estimatorkwargs)
    print('Thetas: ', results['theta'])