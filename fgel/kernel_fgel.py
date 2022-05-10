import cvxpy as cvx
import numpy as np
import torch
import matplotlib.pyplot as plt

from fgel.gel import GeneralizedEL
from fgel.utils.torch_utils import Parameter
from fgel.utils.rkhs_utils import get_rbf_kernel, compute_cholesky_factor

cvx_solver = cvx.MOSEK


class KernelFGEL(GeneralizedEL):

    def __init__(self, kernel_args=None, reg_param=1e-6, **kwargs):
        super().__init__(**kwargs)

        self.reg_param = reg_param

        self.kernel_args = kernel_args
        self.kernel_z = None
        self.k_cholesky = None
        self.kernel_z_val = None

        self.alpha = Parameter(shape=(self.n_sample, self.psi_dim))
        self.set_optimizers(self.alpha)

    def set_kernel(self, z, z_val=None):
        if self.kernel_z is None:
            self.kernel_z = torch.tensor(get_rbf_kernel(z, z, **self.kernel_args), dtype=torch.float32)
            self.k_cholesky = torch.tensor(np.transpose(compute_cholesky_factor(self.kernel_z.detach().numpy())))
        if z_val is not None:
            self.kernel_z_val = get_rbf_kernel(z_val, z_val, **self.kernel_args)

    def compute_alpha_psi(self, x, z):
        return torch.einsum('jr, ji, ir -> i', self.alpha.params, self.kernel_z, self.model.psi(x, z))

    def get_rkhs_norm(self):
        return torch.einsum('ir, ij, jr ->', self.alpha.params, self.kernel_z, self.alpha.params)

    def objective(self, x, z, *args, **kwargs):
        self.set_kernel(z)
        alpha_k_psi = self.compute_alpha_psi(x, z)
        objective = (1/self.n_sample * torch.sum(self.gel_function(alpha_k_psi))
                     - self.reg_param/2 * self.get_rkhs_norm())
        return objective

    def optimize_alpha_cvxpy(self, x_tensor, z_tensor):
        """CVXPY alpha optimization for kernelized objective"""
        self.set_kernel(z_tensor)

        with torch.no_grad():
            try:
                x = [xi.numpy() for xi in x_tensor]
                alpha = cvx.Variable(shape=(self.n_sample, self.psi_dim))
                psi = self.model.psi(x).detach().numpy()
                alpha_psi = np.zeros(self.n_sample)
                for k in range(self.psi_dim):
                    alpha_psi += alpha[:, k] @ self.kernel_z.detach().numpy() @ cvx.diag(psi[:, k])

                objective = (1/self.n_sample * cvx.sum(self.gel_function(alpha_psi, cvxpy=True))
                             - self.reg_param/2 * cvx.square(cvx.norm(cvx.transpose(alpha) @ self.k_cholesky.detach().numpy())))
                if self.divergence_type == 'log':
                    constraint = [alpha_psi <= 1 - self.n_sample]
                else:
                    constraint = []
                problem = cvx.Problem(cvx.Maximize(objective), constraint)
                problem.solve(solver=cvx_solver, verbose=False)
                self.alpha.update_params(alpha.value)
            except:
                print('CVXPY failed. Using old alpha value')
        return

    def init_training(self, x_tensor, z_tensor=None, z_dev_tensor=None):
        self.set_kernel(z_tensor, z_dev_tensor)
        super().init_training(x_tensor, z_tensor, z_dev_tensor)

    def _pretrain_theta(self, x, z, mmr=True):
        if mmr:
            optimizer = torch.optim.LBFGS(self.model.parameters(),
                                          line_search_fn="strong_wolfe")
            def closure():
                optimizer.zero_grad()
                psi = self.model.psi(x, z)
                loss = torch.einsum('ir, ij, jr -> ', psi, self.kernel_z, psi) / (self.n_sample ** 2)
                loss.backward()
                return loss
            optimizer.step(closure)
        else:
            super()._pretrain_theta(x, z)
