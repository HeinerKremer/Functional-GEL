import math
import time
import torch
from fgel.generalized_el import GeneralizedEL
from fgel.utils.oadam import OAdam
from utils.torch_utils import BatchIter, ModularMLPModel


class NeuralFGEL(GeneralizedEL):
    def __init__(self, model, dim_z, reg_param=1e-6,
                 batch_size=None, f_network_kwargs=None, f_optimizer_args=None,
                 **kwargs):

        if f_optimizer_args is None:
            f_optimizer_args = {'lr': 1e-3}

        self.model = model
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.reg_param = reg_param
        self.f_optim_args = f_optimizer_args

        self.f_network_kwargs = self.update_default_f_network_kwargs(f_network_kwargs)
        self.f = ModularMLPModel(**self.f_network_kwargs)
        super().__init__(model=model, **kwargs)

    def set_optimizers(self, param_container=None):
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

    def init_training(self, x_tensor, z_tensor, z_val_tensor=None):
        self.set_kernel(z_tensor, z_val_tensor)
        self.set_optimizers()
        if self.pretrain:
            self._pretrain_theta(x=x_tensor, z=z_tensor)

    def _game_objective(self, x, z):
        fz = self.f(z)
        f_psi = torch.einsum('ik, ik -> i', fz, self.model.psi(x))
        moment = torch.mean(self.gel_function(f_psi))
        if self.reg_param > 0:
            l_reg = self.reg_param / 2 * torch.mean(fz ** 2)
        else:
            l_reg = 0
        return moment, -moment + l_reg

    def _fit_internal(self, x, z, x_val, z_val, debugging=False):
        n = x[0].shape[0]
        if self.batch_size is None:
            self.batch_size = n
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
                x_batch = [x_tensor[0][batch_idx], x_tensor[1][batch_idx]]
                z_batch = z_tensor[batch_idx]
                theta_obj, f_obj = self._game_objective(x_batch, z_batch)

                # update rho network
                self.theta_optimizer.zero_grad()
                theta_obj.backward(retain_graph=True)
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


if __name__ == '__main__':
    from experiments.exp_heteroskedastic import run_heteroskedastic_n_times

    estimatorkwargs = dict(dim_z=1, max_num_epochs=5000, eval_freq=50, divergence='chi2')
    results = run_heteroskedastic_n_times(theta=1.7, noise=1.0, n_train=200, repititions=10,
                                         estimatortype=NeuralFGEL, estimatorkwargs=estimatorkwargs)
    print('Thetas: ', results['theta'])
