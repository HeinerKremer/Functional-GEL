import time
import cvxpy as cvx
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from utils import torch_utils, rkhs_utils
from utils.oadam import OAdam

from fgel.utils.rkhs_utils import get_rbf_kernel
from utils.torch_utils import np_to_tensor
import numpy as np


class AbstractEstimationMethod:
    def __init__(self, model, psi_dim, n_sample=None):
        self.model = model
        self.psi_dim = psi_dim
        self.n_sample = n_sample
        self.is_fit = False

    def fit(self, x, z, x_dev, z_dev, show_plots=False):
        self._fit_internal(x, z, x_dev, z_dev, show_plots=show_plots)
        self.is_fit = True

    def get_trained_parameters(self):
        if not self.is_fit:
            raise RuntimeError("Need to fit model before getting fitted params")
        return self.model.get_parameters()

    def calc_mmr_loss(self, x, kernel_z):
        n = x[0].shape[0]
        if isinstance(x, np.ndarray):
            x_tensor = self._to_tensor(x)
        else:
            x_tensor = x
        psi_m = self.model.eval_psi(x_tensor).detach().cpu().numpy()
        dev_mmr = np.transpose(psi_m) @ kernel_z @ psi_m
        return float(dev_mmr.sum() / (n ** 2))

    def _to_tensor(self, data_array):
        raise np_to_tensor(data_array)

    def _fit_internal(self, x, z, x_dev, z_dev, show_plots):
        raise NotImplementedError()


# class AbstractMomentEstimator(AbstractEstimationMethod):

    # def __init__(self, model, psi_dim, theta_optim_args,
    #              max_num_epochs, eval_freq, max_no_improve=5, burn_in_cycles=5,
    #              pretrain=False, verbose=False,
    #              divergence=None, kernel_args=None, n_sample=None,
    #              outeropt=None, inneropt=None, inneriters=None,):
    #     AbstractEstimationMethod.__init__(self, model, psi_dim, n_sample)
    #     self.divergence_type = divergence
    #     self.divergence = self.set_divergence_function()
    #     self.inneropt = inneropt
    #     self.inneriters = inneriters
    #     self.outeropt = outeropt
    #     self.theta_optim_args = theta_optim_args
    #
    #     self.kernel_args = kernel_args
    #     self.kernel_z = None
    #     self.kernel_z_val = None
    #     self.k_cholesky = None
    #
    #     self.max_num_epochs = max_num_epochs if not self.outeropt == 'lbfgs' else 1
    #     self.eval_freq = eval_freq
    #     self.max_no_improve = max_no_improve
    #     self.burn_in_cycles = burn_in_cycles
    #     self.pretrain = pretrain
    #
    #     self.verbose = verbose
    #
    #     self.alpha_optimizer = None
    #     self.theta_optimizer = None
    #     self.optimize_step = None
    #
    #     self.alpha = torch_utils.Parameter(n_sample=self.n_sample)
    #
    # def set_optimizers(self, param_container):
    #     # Inner optimization settings (alpha)
    #     if self.inneropt == 'adam':
    #         self.alpha_optimizer = torch.optim.Adam(params=self.alpha.parameters(), lr=5e-4, betas=(0.5, 0.9))
    #     elif self.inneropt == 'oadam':
    #         self.alpha_optimizer = OAdam(params=self.alpha.parameters(), lr=5e-4, betas=(0.5, 0.9))
    #     elif self.inneropt == 'lbfgs':
    #         self.alpha_optimizer = torch.optim.LBFGS(param_container.parameters(),
    #                                                  max_iter=500,
    #                                                  line_search_fn="strong_wolfe")
    #     elif self.inneropt == 'cvxpy':
    #         self.alpha_optimizer = None
    #     elif self.inneropt == 'md':
    #         self.alpha_optimizer = None
    #     else:
    #         self.alpha_optimizer = None
    #         # raise NotImplementedError
    #
    #     # Outer optimization settings (theta)
    #     if self.outeropt == 'adam':
    #         self.theta_optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.theta_optim_args["lr"], betas=(0.5, 0.9))
    #         self.optimize_step = self.step_oadam
    #     elif self.outeropt == 'oadam':
    #         self.theta_optimizer = OAdam(params=self.model.parameters(), lr=self.theta_optim_args["lr"], betas=(0.5, 0.9))
    #         self.optimize_step = self.step_oadam
    #     elif self.outeropt == 'lbfgs':
    #         self.theta_optimizer = torch.optim.LBFGS(self.model.parameters(),
    #                                                line_search_fn="strong_wolfe",
    #                                                max_iter=100)
    #         self.optimize_step = self.step_lbfgs
    #     else:
    #         self.alpha_optimizer = None
    #         # raise NotImplementedError
    #
    # def set_divergence_function(self):
    #     if self.divergence_type == 'log':
    #         def divergence(weights=None, cvxpy=False):
    #             if cvxpy:
    #                 return - cvx.sum(cvx.log(self.n_sample * weights))
    #             elif isinstance(weights, np.ndarray):
    #                 return - np.sum(np.log(self.n_sample * weights))
    #             else:
    #                 return - torch.sum(torch.log(self.n_sample * weights))
    #
    #     elif self.divergence_type == 'chi2':
    #         def divergence(weights=None, cvxpy=False):
    #             if cvxpy:
    #                 return cvx.sum_squares(self.n_sample * weights - 1)
    #             elif isinstance(weights, np.ndarray):
    #                 return np.sum(np.square(self.n_sample * weights - 1))
    #             else:
    #                 return torch.sum(torch.square(self.n_sample * weights - 1))
    #     elif self.divergence_type == 'kl':
    #         def divergence(weights=None, cvxpy=False):
    #             if cvxpy:
    #                 return cvx.sum(weights * cvx.log(self.n_sample * weights))
    #             elif isinstance(weights, np.ndarray):
    #                 return np.sum(weights * np.log(self.n_sample * weights))
    #             else:
    #                 return torch.sum(weights * torch.log(self.n_sample * weights))
    #     else:
    #         raise NotImplementedError()
    #     return divergence
    #
    # def set_kernel(self, z, z_val=None):
    #     if self.kernel_z is None:
    #         self.kernel_z = torch.tensor(get_rbf_kernel(z, z, **self.kernel_args), dtype=torch.float32)
    #         self.k_cholesky = torch.tensor(np.transpose(compute_cholesky_factor(self.kernel_z.detach().numpy())))
    #     if z_val is not None:
    #         self.kernel_z_val = get_rbf_kernel(z_val, z_val, **self.kernel_args)
    #
    # def objective(self, x, z, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def _fit_internal(self, x, z, x_dev, z_dev, show_plots):
    #     x_tensor = self._to_tensor(x)
    #     z_tensor = self._to_tensor(z)
    #     x_dev_tensor = self._to_tensor(x_dev)
    #     z_dev_tensor = self._to_tensor(z_dev)
    #
    #     self.alpha.init_params()
    #     self.set_optimizers(self.alpha)
    #
    #     self.set_kernel(z, z_dev)
    #
    #     if self.pretrain:
    #         self._pretrain_psi(x=x_tensor, z=z_tensor)
    #
    #     min_dev_loss = float("inf")
    #     time_0 = time.time()
    #     num_no_improve = 0
    #     cycle_num = 0
    #
    #     losses = []
    #     for epoch_i in range(self.max_num_epochs):
    #         self.psi.train()
    #
    #         obj = self.optimize_step(x_tensor, z_tensor)
    #
    #         if isinstance(obj, torch.Tensor):
    #             obj = obj.detach().numpy()
    #         losses.append(obj)
    #
    #         if epoch_i % self.eval_freq == 0:
    #             cycle_num += 1
    #             dev_loss = self.calc_mmr_loss(x_dev_tensor, self.kernel_z_val)
    #             if self.verbose:
    #                 dev_game_obj = self.objective(x_dev_tensor, z_dev_tensor)
    #                 print("epoch %d, game-obj=%f, def-loss=%f"
    #                       % (epoch_i, dev_game_obj, dev_loss))
    #             if dev_loss < min_dev_loss:
    #                 min_dev_loss = dev_loss
    #                 num_no_improve = 0
    #             elif cycle_num > self.burn_in_cycles:
    #                 num_no_improve += 1
    #             if num_no_improve == self.max_no_improve:
    #                 break
    #     if self.verbose:
    #         print("time taken:", time.time() - time_0)
    #
    #     if show_plots:
    #         fig, ax = plt.subplots(1)
    #         ax.plot(losses[1:])
    #         ax.set_title('')
    #         plt.show()
    #
    # def _pretrain_psi(self, x, z, mmr=True):
    #     optimizer = torch.optim.LBFGS(self.psi.parameters(),
    #                                   line_search_fn="strong_wolfe")
    #     if not mmr:
    #         def closure():
    #             optimizer.zero_grad()
    #             psi_x_z = self.psi(x, z)
    #             loss = (psi_x_z ** 2).mean()
    #             loss.backward()
    #             return loss
    #     else:
    #         def closure():
    #             optimizer.zero_grad()
    #             psi = self.psi(x, z)
    #             loss = torch.einsum('ir, ij, jr -> ', psi, self.kernel_z, psi) / (self.n_sample ** 2)
    #             loss.backward()
    #             return loss
    #
    #     optimizer.step(closure)
    #
    # def _to_tensor(self, data_array):
    #     return np_to_tensor(data_array)
    #
    # def step_oadam(self, x_tensor, z_tensor):
    #     raise NotImplementedError
    #
    # def step_lbfgs(self, x_tensor, z_tensor):
    #     raise NotImplementedError