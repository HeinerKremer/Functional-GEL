import time

import cvxpy as cvx
import numpy as np
import torch
import matplotlib.pyplot as plt

from fgel.abstract_estimation_method import AbstractEstimationMethod
from fgel.utils.oadam import OAdam
from fgel.utils.torch_utils import Parameter

cvx_solver = cvx.MOSEK


class GeneralizedEL(AbstractEstimationMethod):

    def __init__(self, model,
                 max_num_epochs=1000, eval_freq=500, max_no_improve=5, burn_in_cycles=5, pretrain=False, theta_optim_args=None,
                 divergence=None, outeropt='lbfgs', inneropt='lbfgs', inneriters=None, kernel_args=None,
                 verbose=False):
        AbstractEstimationMethod.__init__(self, model, kernel_args)

        if theta_optim_args is None:
            theta_optim_args = {'lr': 5e-2}

        self.divergence_type = divergence
        self.softplus = torch.nn.Softplus(beta=10)
        self.divergence = self.set_divergence_function()
        self.gel_function = self.set_gel_function()

        self.alpha = Parameter(shape=(1, self.psi_dim))

        self.inneropt = inneropt
        self.inneriters = inneriters
        self.outeropt = outeropt
        self.theta_optim_args = theta_optim_args
        self.alpha_optimizer = None
        self.theta_optimizer = None
        self.optimize_step = None
        self.set_optimizers(self.alpha)

        self.max_num_epochs = max_num_epochs if not self.outeropt == 'lbfgs' else 1
        self.eval_freq = eval_freq
        self.max_no_improve = max_no_improve
        self.burn_in_cycles = burn_in_cycles
        self.pretrain = pretrain
        self.verbose = verbose

    def set_divergence_function(self):
        if self.divergence_type == 'log':
            def divergence(weights=None, cvxpy=False):
                n_sample = weights.shape[0]
                if cvxpy:
                    return - cvx.sum(cvx.log(n_sample * weights))
                elif isinstance(weights, np.ndarray):
                    return - np.sum(np.log(n_sample * weights))
                else:
                    return - torch.sum(torch.log(n_sample * weights))

        elif self.divergence_type == 'chi2':
            def divergence(weights=None, cvxpy=False):
                n_sample = weights.shape[0]
                if cvxpy:
                    return cvx.sum_squares(n_sample * weights - 1)
                elif isinstance(weights, np.ndarray):
                    return np.sum(np.square(n_sample * weights - 1))
                else:
                    return torch.sum(torch.square(n_sample * weights - 1))
        elif self.divergence_type == 'kl':
            def divergence(weights=None, cvxpy=False):
                n_sample = weights.shape[0]
                if cvxpy:
                    return cvx.sum(weights * cvx.log(n_sample * weights))
                elif isinstance(weights, np.ndarray):
                    return np.sum(weights * np.log(n_sample * weights))
                else:
                    return torch.sum(weights * torch.log(n_sample * weights))
        else:
            raise NotImplementedError()
        return divergence

    def set_gel_function(self):
        if self.divergence_type == 'log':
            def divergence(x=None, cvxpy=False):
                if not cvxpy:
                    return torch.log(self.softplus(1 - x) + 1)
                else:
                    return cvx.log(1 - x)

        elif self.divergence_type == 'chi2':
            def divergence(x=None, cvxpy=False):
                if not cvxpy:
                    return - torch.square(1/2 * x + 1)  # -1/2 * torch.square(x + 1)
                else:
                    return - cvx.square(1/2 * x + 1)
        elif self.divergence_type == 'kl':
            def divergence(x=None, cvxpy=False):
                if not cvxpy:
                    return - torch.exp(x)
                else:
                    return - cvx.exp(x)
        else:
            raise NotImplementedError
        return divergence

    def set_optimizers(self, param_container):
        # Inner optimization settings (alpha)
        if self.inneropt == 'adam':
            self.alpha_optimizer = torch.optim.Adam(params=param_container.parameters(), lr=5e-4, betas=(0.5, 0.9))
        elif self.inneropt == 'oadam':
            self.alpha_optimizer = OAdam(params=param_container.parameters(), lr=5e-4, betas=(0.5, 0.9))
        elif self.inneropt == 'lbfgs':
            self.alpha_optimizer = torch.optim.LBFGS(param_container.parameters(),
                                                     max_iter=500,
                                                     line_search_fn="strong_wolfe")
        elif self.inneropt == 'cvxpy':
            self.alpha_optimizer = None
        elif self.inneropt == 'md':
            self.alpha_optimizer = None
        else:
            self.alpha_optimizer = None
            # raise NotImplementedError

        # Outer optimization settings (theta)
        if self.outeropt == 'adam':
            self.theta_optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.theta_optim_args["lr"], betas=(0.5, 0.9))
            self.optimize_step = self.gradient_step
        elif self.outeropt == 'oadam':
            self.theta_optimizer = OAdam(params=self.model.parameters(), lr=self.theta_optim_args["lr"], betas=(0.5, 0.9))
            self.optimize_step = self.gradient_step
        elif self.outeropt == 'lbfgs':
            self.theta_optimizer = torch.optim.LBFGS(self.model.parameters(),
                                                   line_search_fn="strong_wolfe",
                                                   max_iter=100)
            self.optimize_step = self.lbfgs_step
        else:
            self.alpha_optimizer = None
            # raise NotImplementedError

    """------------- Methods for standard finite dimensional GEL to be overridden for FGEL ------------"""
    def compute_alpha_psi(self, x, z):
        return self.model.psi(x) @ torch.transpose(self.alpha.params, 1, 0)

    def objective(self, x, z, *args, **kwargs):
        objective = torch.mean(self.gel_function(self.compute_alpha_psi(x, z)))
        return objective

    """--------------------- Optimization methods for theta ---------------------"""
    def gradient_step(self, x_tensor, z_tensor, inneriters=100):
        self.optimize_alpha(x_tensor, z_tensor, iters=inneriters)
        self.theta_optimizer.zero_grad()
        obj = self.objective(x_tensor, z_tensor)
        obj.backward()
        self.theta_optimizer.step()
        return obj

    def lbfgs_step(self, x_tensor, z_tensor):
        losses = []
        def closure():
            self.optimize_alpha(x_tensor, z_tensor)
            if torch.is_grad_enabled():
                self.theta_optimizer.zero_grad()
            obj = self.objective(x_tensor, z_tensor)
            losses.append(obj.detach().numpy())
            if obj.requires_grad:
                obj.backward()
            return obj

        self.theta_optimizer.step(closure)
        return 0

    """--------------------- Optimization methods for alpha ---------------------"""
    def optimize_alpha(self, x_tensor, z_tensor, iters=5000):
        if self.inneropt == 'cvxpy':
            return self.optimize_alpha_cvxpy(x_tensor, z_tensor)
        elif self.inneropt == 'lbfgs':
            return self.optimize_alpha_lbfgs(x_tensor, z_tensor)
        elif self.inneropt == 'adam':
            return self.optimize_alpha_adam(x_tensor, z_tensor, iters=iters)
        else:
            raise NotImplementedError

    def optimize_alpha_cvxpy(self, x_tensor, z_tensor):
        with torch.no_grad():
            x = [xi.numpy() for xi in x_tensor]
            z = z_tensor.numpy()
            n_sample = z.shape[0]

            alpha = cvx.Variable(shape=(1, self.psi_dim))   # (1, k)
            psi = self.model.psi(x).detach().numpy()   # (n_sample, k)
            alpha_psi = psi @ cvx.transpose(alpha)    # (n_sample, 1)

            objective = 1/n_sample * cvx.sum(self.gel_function(alpha_psi, cvxpy=True))
            if self.divergence_type == 'log':
                constraint = [alpha_psi <= 1 - n_sample]
            else:
                constraint = []
            problem = cvx.Problem(cvx.Maximize(objective), constraint)
            problem.solve(solver=cvx_solver, verbose=False)
            self.alpha.update_params(alpha.value)
        return

    def optimize_alpha_lbfgs(self, x_tensor, z_tensor):
        def closure():
            if torch.is_grad_enabled():
                self.alpha_optimizer.zero_grad()
            loss = - self.objective(x_tensor, z_tensor)
            if loss.requires_grad:
                loss.backward()
            return loss

        self.alpha_optimizer.step(closure)
        if np.isnan(np.linalg.norm(self.alpha.params.detach().numpy())):
            with torch.no_grad():
                self.alpha.params.copy_(torch.zeros(self.alpha.shape, dtype=torch.float32))
        return

    def optimize_alpha_adam(self, x_tensor, z_tensor, iters):
        for i in range(iters):
            self.alpha_optimizer.zero_grad()
            obj = - self.objective(x_tensor, z_tensor)
            obj.backward()
            self.alpha_optimizer.step()
            if self.divergence_type == 'log':
                with torch.no_grad():
                    alpha_k_psi = self.compute_alpha_psi(x_tensor, z_tensor)
                    alpha, _ = self.alpha.project_log_input_constraint(alpha_k_psi)
                    self.alpha.update_params(alpha)
            return

    def init_training(self, x_tensor, z_tensor, z_val_tensor=None):
        self.alpha.init_params()
        self.set_optimizers(self.alpha)
        if self.pretrain:
            self._pretrain_theta(x=x_tensor, z=z_tensor)

    def _fit_internal(self, x, z, x_val, z_val, debugging=False):
        x_tensor = self._to_tensor(x)
        z_tensor = self._to_tensor(z)

        self.init_training(x_tensor, z_tensor)

        min_val_loss = float("inf")
        time_0 = time.time()
        num_no_improve = 0
        cycle_num = 0

        losses = []
        for epoch_i in range(self.max_num_epochs):
            self.model.train()

            obj = self.optimize_step(x_tensor, z_tensor)

            if isinstance(obj, torch.Tensor):
                obj = obj.detach().numpy()
            losses.append(obj)

            if epoch_i % self.eval_freq == 0:
                cycle_num += 1
                val_mmr_loss = self.calc_val_mmr(x_val, z_val)
                if self.verbose:
                    val_obj = self.objective(x_val, z_val)
                    print("epoch %d, val-obj=%f, mmr-loss=%f"
                          % (epoch_i, val_obj, val_mmr_loss))
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
            fig, ax = plt.subplots(1)
            ax.plot(losses[1:])
            ax.set_title('')
            plt.show()


    """--------------------- Visualization methods ---------------------"""
    # def compute_weights(self, x, z):
    #     alpha_psi = self.compute_alpha_psi(x, z).detach().numpy()
    #     if self.divergence_type == 'log':
    #         weights = 1 / (1 - alpha_psi) / np.sum(1 / (1 - alpha_psi))
    #     elif self.divergence_type == 'chi2':
    #         weights = (alpha_psi + 1) / np.sum(alpha_psi + 1)
    #     else:
    #         raise NotImplementedError
    #     return weights
    #
    # def compute_profile_likelihood(self, x, z, scenario_class, theta_true, theta_range, plot=True, savepath=None):
    #     x, z = torch.Tensor(x), torch.Tensor(z)
    #     self.set_kernel(z)
    #
    #     profile_divergence = []
    #     for theta in theta_range:
    #         print(theta)
    #         scenario = scenario_class(theta=[theta])
    #         psi_generator = scenario.get_psi_generator()
    #         self.model.psi = psi_generator()
    #         self.model.psi.theta = torch.nn.Parameter(torch.tensor(theta), requires_grad=False)
    #
    #         # Reset states of optimizer and params
    #         self.alpha.init_params()
    #         self.set_optimizers(self.alpha)
    #
    #         self.optimize_alpha(x, z)
    #         profile_divergence.append(self.objective(x, z).detach().numpy())
    #     print(profile_divergence)
    #     if plot:
    #         theta_gel = theta_range[np.argmin(profile_divergence)]
    #         plt.scatter([theta_true], [profile_divergence[np.argmin(np.abs(np.asarray(theta_range) - theta_true))]], s=150, c='r', label=fr'True solution $\theta={theta_true}$')
    #         plt.scatter([theta_gel], [np.min(profile_divergence)], s=150, c='g', label=fr'GEL solution $\theta={theta_gel}$')
    #         plt.legend()
    #         plt.plot(theta_range, profile_divergence)
    #         plt.xlabel(r'$\theta$')
    #         plt.ylabel(r'$R(\theta)$')
    #         plt.title(rf'GEL, n={self.n_sample}')
    #         if savepath:
    #             plt.savefig(savepath)
    #         plt.show()
    #     return profile_divergence
    #
    # def compute_profile_likelihood_2d(self, x, z, scenario_class, theta_true, theta_range1, theta_range2, num_points=25, plot=True, savepath=None):
    #     x, z = self._to_tensor(x), self._to_tensor(z)
    #     self.set_kernel(z)
    #     thetas1 = np.linspace(theta_range1[0], theta_range1[1], num_points)
    #     thetas2 = np.linspace(theta_range2[0], theta_range2[1], num_points)
    #
    #     thetas1grid, thetas2grid = np.meshgrid(thetas1, thetas2)
    #
    #     profile_divergence = np.empty([len(thetas1), len(thetas2)])
    #     for i, theta1 in enumerate(thetas1):
    #         for j, theta2 in enumerate(thetas2):
    #             theta = [theta1, theta2]
    #             print(theta)
    #             scenario = scenario_class(theta=theta)
    #             psi_generator = scenario.get_psi_generator()
    #             self.model.psi = psi_generator()
    #             self.model.psi.theta = torch.nn.Parameter(torch.tensor(theta), requires_grad=False)
    #
    #             # Reset states of optimizer and params
    #             self.alpha.init_params()
    #             self.set_optimizers(self.alpha)
    #
    #             self.optimize_alpha(x, z)
    #             profile_divergence[i, j] = self.objective(x, z).detach().numpy()
    #     print(profile_divergence)
    #     if plot:
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)#, projection='3d')
    #         ax.imshow(profile_divergence, aspect='auto', extent=([theta_range1[0], theta_range1[1], theta_range2[0], theta_range2[1]]))
    #         plt.show()
    #     return profile_divergence

if __name__ == '__main__':
    from experiments.exp_heteroskedastic import run_heteroskedastic_n_times

    estimatorkwargs = dict(theta_optim_args={}, max_num_epochs=100, eval_freq=50,
                           divergence='chi2', outeropt='lbfgs', inneropt='cvxpy', inneriters=100)
    results = run_heteroskedastic_n_times(theta=1.7, noise=1.0, n_train=200, repititions=10,
                                         estimatortype=GeneralizedEL, estimatorkwargs=estimatorkwargs)
    print('Thetas: ', results['theta'])