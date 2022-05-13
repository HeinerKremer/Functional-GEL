import cvxpy as cvx
import numpy as np
import torch
import matplotlib.pyplot as plt

from fgel.generalized_el import GeneralizedEL
from fgel.utils.torch_utils import Parameter
from fgel.utils.rkhs_utils import get_rbf_kernel, compute_cholesky_factor

cvx_solver = cvx.MOSEK


class KernelFGEL(GeneralizedEL):

    def __init__(self, reg_param=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.reg_param = reg_param

        # For KernelFGEL alpha depends on sample size and will be initialized together with the kernel
        self.alpha = None

    def set_kernel(self, z, z_val=None):
        super().set_kernel(z=z, z_val=z_val)
        if self.alpha is None:
            self.alpha = Parameter(shape=(self.kernel_z.shape[0], self.psi_dim))

    def init_training(self, x_tensor, z_tensor=None, z_val_tensor=None):
        self.set_kernel(z_tensor, z_val_tensor)
        super().init_training(x_tensor, z_tensor)

    def compute_alpha_psi(self, x, z):
        return torch.einsum('jr, ji, ir -> i', self.alpha.params, self.kernel_z, self.model.psi(x))

    def get_rkhs_norm(self):
        return torch.einsum('ir, ij, jr ->', self.alpha.params, self.kernel_z, self.alpha.params)

    def objective(self, x, z, *args, **kwargs):
        self.set_kernel(z)
        alpha_k_psi = self.compute_alpha_psi(x, z)
        objective = torch.mean(self.gel_function(alpha_k_psi)) - self.reg_param/2 * self.get_rkhs_norm()
        return objective

    def optimize_alpha_cvxpy(self, x_tensor, z_tensor):
        """CVXPY alpha optimization for kernelized objective"""
        n_sample = z_tensor.shape[0]
        self.set_kernel(z_tensor)

        with torch.no_grad():
            try:
                x = [xi.numpy() for xi in x_tensor]
                alpha = cvx.Variable(shape=(n_sample, self.psi_dim))
                psi = self.model.psi(x).detach().numpy()
                alpha_psi = np.zeros(n_sample)
                for k in range(self.psi_dim):
                    alpha_psi += alpha[:, k] @ self.kernel_z.detach().numpy() @ cvx.diag(psi[:, k])

                objective = (1/n_sample * cvx.sum(self.gel_function(alpha_psi, cvxpy=True))
                             - self.reg_param/2 * cvx.square(cvx.norm(cvx.transpose(alpha) @ self.k_cholesky.detach().numpy())))
                if self.divergence_type == 'log':
                    constraint = [alpha_psi <= 1 - n_sample]
                else:
                    constraint = []
                problem = cvx.Problem(cvx.Maximize(objective), constraint)
                problem.solve(solver=cvx_solver, verbose=False)
                self.alpha.update_params(alpha.value)
            except:
                print('CVXPY failed. Using old alpha value')
        return


if __name__ == '__main__':
    from experiments.exp_heteroskedastic import run_heteroskedastic_n_times

    estimatorkwargs = dict(theta_optim_args={}, max_num_epochs=100, eval_freq=50,
                           divergence='chi2', outeropt='lbfgs', inneropt='lbfgs', inneriters=100)
    results = run_heteroskedastic_n_times(theta=1.7, noise=1.0, n_train=200, repititions=20,
                                         estimatortype=KernelFGEL, estimatorkwargs=estimatorkwargs)
    print('Thetas: ', results['theta'])