import time

import cvxpy as cvx
import numpy as np
import torch
import matplotlib.pyplot as plt

from fgel.utils.rkhs_utils import get_rbf_kernel
from fgel.utils.torch_utils import Parameter, BatchIter
from fgel.generalized_el import GeneralizedEL

cvx_solver = cvx.MOSEK


class MMDGEL(GeneralizedEL):

    def __init__(self, kernel_x_kwargs=None, kl_reg_param=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.kl_reg_param = kl_reg_param
        self.kernel_x_kwargs = kernel_x_kwargs
        self.kernel_x = None

    def _set_kernel_x(self, x, x_val=None):
        # Use product kernels for now (TAKE CARE WHEN IMPLEMENTING SOMETHING WITH ONLY T NO Y)
        if self.kernel_x is None and x is not None:
            self.kernel_x = (get_rbf_kernel(x[0], x[0], **self.kernel_x_kwargs).type(torch.float32)
                             * get_rbf_kernel(x[1], x[1], **self.kernel_x_kwargs).type(torch.float32))
        if x_val is not None:
            self.kernel_x_val = (get_rbf_kernel(x_val[0], x_val[0], **self.kernel_x_kwargs)
                                 * get_rbf_kernel(x[1], x[1], **self.kernel_x_kwargs).type(torch.float32))

    def _init_dual_func(self):
        self.dual_func = Parameter(shape=(1, self.psi_dim))
        self.rkhs_func = Parameter(shape=(self.kernel_z.shape[0], 1))
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

            dual_func = cvx.Variable(shape=(1, self.psi_dim))   # (1, k)
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
        self._set_kernel(z_tensor, z_val_tensor)
        self._set_kernel_x(x_tensor, x_val_tensor)
        self._init_dual_func()
        self._set_optimizers()
        if self.pretrain:
            self._pretrain_theta(x=x_tensor, z=z_tensor)

    def _train_internal(self, x_train, z_train, x_val, z_val, debugging):
        x_tensor = self._to_tensor(x_train)
        z_tensor = self._to_tensor(z_train)
        x_val_tensor = self._to_tensor(x_val)
        z_val_tensor = self._to_tensor(z_val)

        if self.batch_training:
            n = z_train.shape[0]
            batch_iter = BatchIter(num=n, batch_size=self.batch_size)
            batches_per_epoch = np.ceil(n / self.batch_size)
            eval_freq_epochs = np.ceil(self.eval_freq / batches_per_epoch)
        else:
            eval_freq_epochs = self.eval_freq

        self._init_training(x_tensor, z_tensor)
        # loss = []
        mmr = []

        min_val_loss = float("inf")
        time_0 = time.time()
        num_no_improve = 0
        cycle_num = 0

        for epoch_i in range(self.max_num_epochs):
            self.model.train()
            self.dual_func.train()
            if self.batch_training:
                for batch_idx in batch_iter:
                    x_batch = [x_tensor[0][batch_idx], x_tensor[1][batch_idx]]
                    z_batch = z_tensor[batch_idx]
                    obj = self.optimize_step(x_batch, z_batch)
                    # loss.append(obj)
            else:
                obj = self.optimize_step(x_tensor, z_tensor)
                # loss.append(obj)

            if epoch_i % eval_freq_epochs == 0:
                cycle_num += 1
                val_mmr_loss = self._calc_val_mmr(x_val, z_val)
                if self.verbose:
                    val_theta_obj, _ = self.objective(x_val_tensor, z_val_tensor)
                    print("epoch %d, theta-obj=%f, val-mmr-loss=%f"
                          % (epoch_i, val_theta_obj, val_mmr_loss))
                mmr.append(float(val_mmr_loss))
                if val_mmr_loss < min_val_loss:
                    min_val_loss = val_mmr_loss
                    num_no_improve = 0
                elif cycle_num > self.burn_in_cycles:
                    num_no_improve += 1
                if num_no_improve == self.max_no_improve:
                    break
        if self.verbose:
            print("time taken:", time.time() - time_0)
        if debugging:
            try:
                plt.plot(mmr)
                plt.show()
            except:
                pass



if __name__ == '__main__':
    from experiments.exp_heteroskedastic import run_heteroskedastic_n_times

    estimatorkwargs = dict(theta_optim_args={}, max_num_epochs=100, eval_freq=50,
                           divergence='chi2', outeropt='lbfgs', inneropt='cvxpy', inneriters=100)
    results = run_heteroskedastic_n_times(theta=1.7, noise=1.0, n_train=200, repititions=10,
                                          estimatortype=GeneralizedEL, estimatorkwargs=estimatorkwargs)
    print('Thetas: ', results['theta'])
