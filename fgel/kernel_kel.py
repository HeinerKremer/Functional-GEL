import cvxpy as cvx
import numpy as np
import torch

from fgel.kel import KernelEL
from fgel.utils.rkhs_utils import get_rbf_kernel, compute_cholesky_factor
from fgel.utils.torch_utils import Parameter

cvx_solver = cvx.MOSEK


class KernelELKernel(KernelEL):

    def __init__(self, reg_param, **kwargs):
        super().__init__(**kwargs)
        self.reg_param = reg_param

    def _init_dual_params(self):
        self.dual_moment_func = Parameter(shape=(self.kernel_z.shape[0], self.dim_psi))
        self.rkhs_func = Parameter(shape=(self.kernel_x.shape[0], 1))
        self.dual_normalization = Parameter(shape=(1, 1))
        self.all_dual_params = list(self.dual_moment_func.parameters()) + list(self.dual_normalization.parameters()) + list(self.rkhs_func.parameters())

    """------------- Objective of Kernel-EL-Kernel ------------"""
    def eval_dual_moment_func(self, z):
        return torch.einsum('ij, ik -> kj', self.dual_moment_func.params, self.kernel_z)

    def objective(self, x, z, *args, **kwargs):
        objective, _ = super().objective(x, z, *args, **kwargs)
        regularizer = self.reg_param/2 * torch.einsum('ir, ij, jr ->', self.dual_moment_func.params,
                                                      self.kernel_z, self.dual_moment_func.params)
        return objective, -objective + regularizer


if __name__ == '__main__':
    from experiments.tests import test_cmr_estimator
    test_cmr_estimator(estimation_method='KernelELKernel', n_runs=1, n_train=1000, hyperparams=None)
