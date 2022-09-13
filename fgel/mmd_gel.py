import cvxpy as cvx
import numpy as np
import torch

from fgel.utils.rkhs_utils import get_rbf_kernel
from fgel.utils.torch_utils import Parameter
from fgel.generalized_el import GeneralizedEL

cvx_solver = cvx.MOSEK


class MMDEL(GeneralizedEL):
    """
    Maximum mean discrepancy empirical likelihood estimator for unconditional moment restrictions.
    """

    def __init__(self, kl_reg_param=1e-6, kernel_x_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.kl_reg_param = kl_reg_param

        if kernel_x_kwargs is None:
            kernel_x_kwargs = {}
        self.kernel_x_kwargs = kernel_x_kwargs
        self.kernel_x = None
        self.kernel_x_val = None

    def _set_kernel_x(self, x, x_val=None):
        # Use product kernels for now (TAKE CARE WHEN IMPLEMENTING SOMETHING WITH ONLY T NO Y)
        if self.kernel_x is None and x is not None:
            self.kernel_x = (get_rbf_kernel(x[0], x[0], **self.kernel_x_kwargs).type(torch.float32)
                             * get_rbf_kernel(x[1], x[1], **self.kernel_x_kwargs).type(torch.float32))
        if x_val is not None:
            self.kernel_x_val = (get_rbf_kernel(x_val[0], x_val[0], **self.kernel_x_kwargs)
                                 * get_rbf_kernel(x[1], x[1], **self.kernel_x_kwargs).type(torch.float32))

    def _init_dual_func(self):
        self.dual_func = Parameter(shape=(1, self.dim_psi))
        self.rkhs_func = Parameter(shape=(self.kernel_x.shape[0], 1))
        self.dual_normalization = Parameter(shape=(1, 1))

    def _set_divergence_function(self):
        def divergence(weights=None, cvxpy=False):
            raise NotImplementedError('MMD divergence non-trivial to compute here')
        return divergence

    def _set_gel_function(self):
        def gel_function():
            raise NotImplementedError('gel_function not used for MMD-GEL')
        return gel_function

    def _set_optimizers(self):
        dual_params = list(self.dual_func.parameters()) + list(self.dual_normalization.parameters()) + list(self.rkhs_func.parameters())
        super()._set_optimizers(dual_params=dual_params)

    """------------- Objective of MMD-GEL ------------"""
    def objective(self, x, z, *args, **kwargs):
        expected_rkhs_func = torch.mean(torch.einsum('ij, ik -> k', self.rkhs_func.params, self.kernel_x))
        rkhs_norm_sq = torch.einsum('ir, ij, jr ->', self.rkhs_func.params, self.kernel_x, self.rkhs_func.params)
        exponent = (torch.einsum('ij, ik -> k', self.rkhs_func.params, self.kernel_x) + self.dual_normalization.params
                    - torch.einsum('ij, ij -> i', self.dual_func.params, self.model.psi(x)))
        objective = (expected_rkhs_func + self.dual_normalization.params - 1 / 2 * rkhs_norm_sq
                     - self.kl_reg_param * torch.mean(torch.exp(1 / self.kl_reg_param * exponent)))
        return objective, -objective

    """--------------------- Optimization methods for dual_func ---------------------"""
    def _optimize_dual_func_cvxpy(self, x_tensor, z_tensor):
        with torch.no_grad():
            x = [xi.numpy() for xi in x_tensor]
            z = z_tensor.numpy()
            n_sample = z.shape[0]

            dual_func = cvx.Variable(shape=(1, self.dim_psi))   # (1, k)
            dual_normalization = cvx.Variable(shape=(1, 1))
            rkhs_func = cvx.Variable(shape=(n_sample, 1))

            kernel_x = self.kernel_x.detach().numpy()
            psi = self.model.psi(x).detach().numpy()   # (n_sample, k)

            dual_func_psi = psi @ cvx.transpose(dual_func)    # (n_sample, 1)
            expected_rkhs_func = 1/n_sample * cvx.sum(kernel_x @ rkhs_func)
            rkhs_norm_sq = cvx.sum(cvx.transpose(rkhs_func) @ kernel_x @ rkhs_func)
            exponent = cvx.sum(kernel_x @ rkhs_func + dual_normalization - dual_func_psi, axis=1)
            objective = (expected_rkhs_func + dual_normalization - 1 / 2 * rkhs_norm_sq
                         - self.kl_reg_param / n_sample * cvx.sum(cvx.exp(1 / self.kl_reg_param * exponent)))

            problem = cvx.Problem(cvx.Maximize(objective))
            problem.solve(solver=cvx_solver, verbose=False)

            self.dual_func.update_params(dual_func.value)
            self.rkhs_func.update_params(rkhs_func.value)
            self.dual_normalization.update_params(dual_normalization.value)
        return

    def _init_training(self, x_tensor, z_tensor, x_val_tensor=None, z_val_tensor=None):
        self._set_kernel_z(z_tensor, z_val_tensor)
        self._set_kernel_x(x_tensor, x_val_tensor)
        self._init_dual_func()
        self._set_optimizers()
        if self.pretrain:
            self._pretrain_theta(x=x_tensor, z=z_tensor)


if __name__ == '__main__':

    kl_reg_param = 1e-3

    estimator_kwargs = {
        "dual_optim": 'lbfgs',
        "theta_optim": 'lbfgs',
        "eval_freq": 100,
        "max_num_epochs": 20000,
    }

    exp = HeteroskedasticNoiseExperiment(theta=[theta], noise=noise, heteroskedastic=True)
    exp.prepare_dataset(n_train=100, n_val=100, n_test=20000)
    model = exp.init_model()

    estimator = MMDEL(model=model, kl_reg_param=kl_reg_param, **estimator_kwargs)

