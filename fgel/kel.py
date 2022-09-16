import cvxpy as cvx
import numpy as np
import torch

from fgel.utils.rkhs_utils import get_rbf_kernel, compute_cholesky_factor
from fgel.utils.torch_utils import Parameter
from fgel.generalized_el import GeneralizedEL

cvx_solver = cvx.MOSEK


class KernelEL(GeneralizedEL):
    """
    Maximum mean discrepancy empirical likelihood estimator for unconditional moment restrictions.
    """

    def __init__(self, kl_reg_param, kernel_x_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.kl_reg_param = kl_reg_param

        if kernel_x_kwargs is None:
            kernel_x_kwargs = {}
        self.kernel_x_kwargs = kernel_x_kwargs
        self.kernel_x = None
        self.kernel_x_val = None

    def _set_kernel_x(self, x, x_val=None):
        if self.kernel_x is None and x is not None:
            self.kernel_x = (get_rbf_kernel(x[0], x[0], **self.kernel_x_kwargs).type(torch.float32)
                             * get_rbf_kernel(x[1], x[1], **self.kernel_x_kwargs).type(torch.float32))
            k_cholesky = torch.tensor(np.transpose(compute_cholesky_factor(self.kernel_x.detach().numpy())))
            self.kernel_x_cholesky = k_cholesky

        if x_val is not None:
            self.kernel_x_val = (get_rbf_kernel(x_val[0], x_val[0], **self.kernel_x_kwargs)
                                 * get_rbf_kernel(x_val[1], x_val[1], **self.kernel_x_kwargs).type(torch.float32))

    def _init_dual_params(self):
        self.dual_moment_func = Parameter(shape=(1, self.dim_psi))
        self.rkhs_func = Parameter(shape=(self.kernel_x.shape[0], 1))
        self.dual_normalization = Parameter(shape=(1, 1))
        self.all_dual_params = list(self.dual_moment_func.parameters()) + list(self.dual_normalization.parameters()) + list(self.rkhs_func.parameters())

    def _set_divergence_function(self):
        def divergence(weights=None, cvxpy=False):
            raise NotImplementedError('MMD divergence non-trivial to compute here')
        return divergence

    def _set_gel_function(self):
        def gel_function():
            raise NotImplementedError('gel_function not used for MMD-GEL')
        return gel_function

    """------------- Objective of MMD-GEL ------------"""
    def eval_dual_moment_func(self, z):
        return self.dual_moment_func.params

    def objective(self, x, z, *args, **kwargs):
        expected_rkhs_func = torch.mean(torch.einsum('ij, ik -> k', self.rkhs_func.params, self.kernel_x))
        rkhs_norm_sq = torch.einsum('ir, ij, jr ->', self.rkhs_func.params, self.kernel_x, self.rkhs_func.params)
        exponent = (torch.einsum('ij, ik -> k', self.rkhs_func.params, self.kernel_x) + self.dual_normalization.params
                    - torch.einsum('ik, ik -> i', self.eval_dual_moment_func(z), self.model.psi(x)))
        objective = (expected_rkhs_func + self.dual_normalization.params - 1 / 2 * rkhs_norm_sq
                     - self.kl_reg_param * torch.mean(torch.exp(1 / self.kl_reg_param * exponent)))
        return objective, -objective

    """--------------------- Optimization methods for dual_func ---------------------"""
    def _optimize_dual_func_cvxpy(self, x_tensor, z_tensor):
        with torch.no_grad():
            x = [xi.numpy() for xi in x_tensor]
            n_sample = x[0].shape[0]

            dual_func = cvx.Variable(shape=(1, self.dim_psi))   # (1, k)
            dual_normalization = cvx.Variable(shape=(1, 1))
            rkhs_func = cvx.Variable(shape=(n_sample, 1))

            kernel_x = self.kernel_x.detach().numpy()
            psi = self.model.psi(x).detach().numpy()   # (n_sample, k)

            dual_func_psi = psi @ cvx.transpose(dual_func)    # (n_sample, 1)
            expected_rkhs_func = 1/n_sample * cvx.sum(kernel_x @ rkhs_func)
            rkhs_norm_sq = cvx.square(cvx.norm(cvx.transpose(rkhs_func) @ self.kernel_x_cholesky.detach().numpy())) #cvx.quad_form(rkhs_func, kernel_x)
            objective = (expected_rkhs_func + dual_normalization - 1 / 2 * rkhs_norm_sq)

            exponent = cvx.sum(kernel_x @ rkhs_func + dual_normalization - dual_func_psi, axis=1)
            objective += - self.kl_reg_param / n_sample * cvx.sum(cvx.exp(1 / self.kl_reg_param * exponent))

            problem = cvx.Problem(cvx.Maximize(objective))
            problem.solve(solver=cvx_solver, verbose=True)

            if dual_normalization.value is None or dual_func.value is None or rkhs_func.value is None:
                raise RuntimeError('Dual parameter optimization failed.')

            self.dual_moment_func.update_params(dual_func.value)
            self.rkhs_func.update_params(rkhs_func.value)
            self.dual_normalization.update_params(dual_normalization.value)
        return

    def _init_training(self, x_tensor, z_tensor, x_val_tensor=None, z_val_tensor=None):
        self._set_kernel_z(z_tensor, z_val_tensor)
        self._set_kernel_x(x_tensor, x_val_tensor)
        self._init_dual_params()
        self._set_optimizers()
        if self.pretrain:
            self._pretrain_theta(x=x_tensor, z=z_tensor)


if __name__ == '__main__':
    from experiments.tests import test_mr_estimator
    test_mr_estimator(estimation_method='KernelEL', n_runs=5, n_train=2000, hyperparams=None)