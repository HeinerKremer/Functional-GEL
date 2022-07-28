import cvxpy as cvx
import numpy as np
import torch

from fgel.generalized_el import GeneralizedEL
from fgel.utils.torch_utils import Parameter

cvx_solver = cvx.MOSEK


class KernelFGEL(GeneralizedEL):

    def __init__(self, reg_param=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.reg_param = reg_param

    def _init_dual_func(self):
        self.dual_func = Parameter(shape=(self.kernel_z.shape[0], self.psi_dim))
        # self.dual_normalization = Parameter(shape=(1, 1))
        self.params = torch.nn.Parameter(torch.zeros(size=(1, 1), dtype=torch.float32), requires_grad=True)

    def get_rkhs_norm(self):
        return torch.einsum('ir, ij, jr ->', self.dual_func.params, self.kernel_z, self.dual_func.params)

    def objective(self, x, z, *args, **kwargs):
        dual_func_k_psi = torch.einsum('jr, ji, ir -> i', self.dual_func.params, self.kernel_z, self.model.psi(x))
        objective = torch.mean(self.gel_function(dual_func_k_psi))
        regularizer = self.reg_param/2 * torch.sqrt(self.get_rkhs_norm())
        return objective, - objective + regularizer

    # def objective(self, x, z, *args, **kwargs):
    #     dual_func_k_psi = torch.einsum('jr, ji, ir -> i', self.dual_func.params, self.kernel_z, self.model.psi(x))
    #     objective = torch.mean(self.gel_function(torch.squeeze(self.dual_normalization.params) + dual_func_k_psi))
    #     regularizer = self.reg_param/2 * torch.sqrt(self.get_rkhs_norm())
    #     # print(self.dual_normalization.params.detach().numpy(), (- objective + regularizer).detach().numpy())
    #     return objective, - objective + regularizer - self.dual_normalization.params

    def _optimize_dual_func_cvxpy(self, x_tensor, z_tensor):
        """CVXPY dual_func optimization for kernelized objective"""
        n_sample = z_tensor.shape[0]
        self._set_kernel(z_tensor)

        with torch.no_grad():
            try:
                x = [xi.numpy() for xi in x_tensor]
                dual_func = cvx.Variable(shape=(n_sample, self.psi_dim))
                psi = self.model.psi(x).detach().numpy()
                dual_func_psi = np.zeros(n_sample)
                for k in range(self.psi_dim):
                    dual_func_psi += dual_func[:, k] @ self.kernel_z.detach().numpy() @ cvx.diag(psi[:, k])

                objective = (1/n_sample * cvx.sum(self.gel_function(dual_func_psi, cvxpy=True))
                             - self.reg_param/2 * cvx.square(cvx.norm(cvx.transpose(dual_func) @ self.k_cholesky.detach().numpy())))
                if self.divergence_type == 'log':
                    constraint = [dual_func_psi <= 1 - n_sample]
                else:
                    constraint = []
                problem = cvx.Problem(cvx.Maximize(objective), constraint)
                problem.solve(solver=cvx_solver, verbose=False)
                self.dual_func.update_params(dual_func.value)
            except:
                print('CVXPY failed. Using old dual_func value')
        return


if __name__ == '__main__':
    from experiments.exp_heteroskedastic import run_heteroskedastic_n_times

    estimatorkwargs = dict(divergence='log', theta_optim='lbfgs', dual_optim='lbfgs',
                           max_num_epochs=50000, eval_freq=2000, pretrain=True)
    results = run_heteroskedastic_n_times(theta=1.7, noise=1.0, n_train=200, repititions=10,
                                          estimatortype=KernelFGEL, estimatorkwargs=estimatorkwargs)
    print('Thetas: ', results['theta'])
    print('Train risk: ', results['train_risk'])
