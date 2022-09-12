import time

import cvxpy as cvx
import numpy as np
import torch
import matplotlib.pyplot as plt

from fgel.abstract_estimation_method import AbstractEstimationMethod
from fgel.utils.oadam import OAdam
from fgel.utils.torch_utils import Parameter, BatchIter

cvx_solver = cvx.MOSEK


class GeneralizedEL(AbstractEstimationMethod):
    """
    The standard f-divergence based generalized empirical likelihood estimator of Owen and Qin and Lawless for
    unconditional moment restrictions. This is the base class that all GEL-based estimators inherit from.
    Optimization procedures and general functionalities should be implemented here. The child classes should usually only
    override the `objective`, `_init_dual_func`, `_init_training` methods and include methods for computing specific
    quantities (and if desired a cvxpy optimization method for the optimization over the dual functions).
    """

    def __init__(self, model,
                 max_num_epochs=50000, eval_freq=2000, max_no_improve=3, burn_in_cycles=5,
                 theta_optim=None, theta_optim_args=None, pretrain=True,
                 dual_optim=None, dual_optim_args=None, inneriters=None,
                 divergence=None, kernel_z_kwargs=None,
                 verbose=False):
        super().__init__(model=model, kernel_z_kwargs=kernel_z_kwargs)

        if theta_optim_args is None:
            theta_optim_args = {"lr": 5e-4}

        if dual_optim_args is None:
            dual_optim_args = {"lr": 5 * 5e-4}

        self.divergence_type = divergence
        self.softplus = torch.nn.Softplus(beta=10)
        self.divergence = self._set_divergence_function()
        self.gel_function = self._set_gel_function()

        self.dual_func = None
        self.dual_normalization = None
        self.dual_optim_type = dual_optim
        self.dual_func_optim_args = dual_optim_args
        self.dual_func_optimizer = None
        self.inneriters = inneriters

        self.theta_optim_type = theta_optim
        self.theta_optim_args = theta_optim_args
        self.theta_optimizer = None

        self.optimize_step = None

        self.max_num_epochs = max_num_epochs if not self.theta_optim_type == 'lbfgs' else 3
        self.eval_freq = eval_freq
        self.max_no_improve = max_no_improve
        self.burn_in_cycles = burn_in_cycles
        self.pretrain = pretrain
        self.batch_training = False
        self.batch_size = None
        self.verbose = verbose

    def _init_dual_func(self):
        self.dual_func = Parameter(shape=(1, self.psi_dim))

    def _set_divergence_function(self):
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
        elif self.divergence_type == 'off':
            return None
        else:
            raise NotImplementedError()
        return divergence

    def _set_gel_function(self):
        if self.divergence_type == 'log':
            def gel_function(x=None, cvxpy=False):
                if not cvxpy:
                    return torch.log(self.softplus(1 - x) + 1 / x.shape[0])
                else:
                    return cvx.log(1 - x)

        elif self.divergence_type == 'chi2':
            def gel_function(x=None, cvxpy=False):
                if not cvxpy:
                    return - 1/2 * torch.square(x + 1)  # -1/2 * torch.square(x + 1)
                else:
                    return - cvx.square(1/2 * x + 1)
        elif self.divergence_type == 'kl':
            def gel_function(x=None, cvxpy=False):
                if not cvxpy:
                    return - torch.exp(x)
                else:
                    return - cvx.exp(x)
        elif self.divergence_type == 'off':
            return None
        else:
            raise NotImplementedError
        return gel_function

    def _set_theta_optimizer(self):
        # Outer optimization settings (theta)
        if self.theta_optim_type == 'adam':
            self.theta_optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.theta_optim_args["lr"],
                                                    betas=(0.5, 0.9))
            self.optimize_step = self._gradient_step
        elif self.theta_optim_type == 'oadam':
            self.theta_optimizer = OAdam(params=self.model.parameters(), lr=self.theta_optim_args["lr"],
                                         betas=(0.5, 0.9))
            self.optimize_step = self._gradient_step
        elif self.theta_optim_type == 'lbfgs':
            self.theta_optimizer = torch.optim.LBFGS(self.model.parameters(),
                                                     line_search_fn="strong_wolfe",
                                                     max_iter=100)
            self.optimize_step = self._lbfgs_step
        if self.theta_optim_type == 'oadam_gda':
            # Optimistic Adam gradient descent ascent (e.g. for neural FGEL/VMM)
            self.theta_optimizer = OAdam(params=self.model.parameters(), lr=self.theta_optim_args["lr"],
                                         betas=(0.5, 0.9))
            self.dual_optim_type = 'oadam_gda'
            self._set_dual_optimizer()
            self.optimize_step = self._gradient_descent_ascent_step
        else:
            raise NotImplementedError('Invalid `theta` optimizer specified.')

    def _set_dual_optimizer(self, dual_params=None):
        if dual_params is None:
            dual_params = self.dual_func.parameters()

        # Inner optimization settings (dual_func)
        if self.dual_optim_type == 'adam':
            self.dual_func_optimizer = torch.optim.Adam(params=dual_params,
                                                        lr=self.dual_func_optim_args["lr"], betas=(0.5, 0.9))
        elif self.dual_optim_type == 'oadam':
            self.dual_func_optimizer = OAdam(params=dual_params,
                                             lr=self.dual_func_optim_args["lr"], betas=(0.5, 0.9))
        elif self.dual_optim_type == 'lbfgs':
            self.dual_func_optimizer = torch.optim.LBFGS(dual_params,
                                                             max_iter=500,
                                                             line_search_fn="strong_wolfe")
        else:
            self.dual_func_optimizer = None

    def _set_optimizers(self, dual_params=None):
        self._set_dual_optimizer(dual_params=dual_params)
        self._set_theta_optimizer()

    """------------- Objective of standard finite dimensional GEL to be overridden for FGEL ------------"""
    def objective(self, x, z, *args, **kwargs):
        dual_func_psi = self.model.psi(x) @ torch.transpose(self.dual_func.params, 1, 0)
        objective = torch.mean(self.gel_function(dual_func_psi))
        return objective, -objective

    """--------------------- Optimization methods for theta ---------------------"""
    def _gradient_step(self, x_tensor, z_tensor, inneriters=100):
        self.optimize_dual_func(x_tensor, z_tensor, iters=inneriters)
        self.theta_optimizer.zero_grad()
        obj, _ = self.objective(x_tensor, z_tensor)
        obj.backward()
        self.theta_optimizer.step()
        return float(obj.detach().numpy())

    def _lbfgs_step(self, x_tensor, z_tensor):
        losses = []

        def closure():
            self.optimize_dual_func(x_tensor, z_tensor)
            if torch.is_grad_enabled():
                self.theta_optimizer.zero_grad()
            obj, _ = self.objective(x_tensor, z_tensor)
            losses.append(obj)
            if obj.requires_grad:
                obj.backward()
            return obj

        self.theta_optimizer.step(closure)
        # print(self.theta_optimizer.state_dict())
        return [float(loss.detach().numpy()) for loss in losses]

    def _gradient_descent_ascent_step(self, x_tensor, z_tensor):
        theta_obj, dual_func_obj = self.objective(x_tensor, z_tensor)

        # update theta
        self.theta_optimizer.zero_grad()
        theta_obj.backward(retain_graph=True)
        # theta_obj.backward()
        self.theta_optimizer.step()

        # update dual function
        self.dual_func_optimizer.zero_grad()
        dual_func_obj.backward()
        self.dual_func_optimizer.step()
        return float(- dual_func_obj.detach().numpy())

    """--------------------- Optimization methods for dual_func ---------------------"""
    def optimize_dual_func(self, x_tensor, z_tensor, iters=5000):
        if self.dual_optim_type == 'cvxpy':
            return self._optimize_dual_func_cvxpy(x_tensor, z_tensor)
        elif self.dual_optim_type == 'lbfgs':
            return self._optimize_dual_func_lbfgs(x_tensor, z_tensor)
        elif self.dual_optim_type == 'adam' or self.dual_optim_type == 'oadam':
            return self._optimize_dual_func_gd(x_tensor, z_tensor, iters=iters)
        else:
            raise NotImplementedError

    def _optimize_dual_func_cvxpy(self, x_tensor, z_tensor):
        with torch.no_grad():
            x = [xi.numpy() for xi in x_tensor]
            z = z_tensor.numpy()
            n_sample = z.shape[0]

            dual_func = cvx.Variable(shape=(1, self.psi_dim))   # (1, k)
            psi = self.model.psi(x).detach().numpy()   # (n_sample, k)
            dual_func_psi = psi @ cvx.transpose(dual_func)    # (n_sample, 1)

            objective = 1/n_sample * cvx.sum(self.gel_function(dual_func_psi, cvxpy=True))
            if self.divergence_type == 'log':
                constraint = [dual_func_psi <= 1 - n_sample]
            else:
                constraint = []
            problem = cvx.Problem(cvx.Maximize(objective), constraint)
            problem.solve(solver=cvx_solver, verbose=False)
            self.dual_func.update_params(dual_func.value)
        return

    def _optimize_dual_func_lbfgs(self, x_tensor, z_tensor):
        def closure():
            if torch.is_grad_enabled():
                self.dual_func_optimizer.zero_grad()
            _, loss_dual_func = self.objective(x_tensor, z_tensor)
            if loss_dual_func.requires_grad:
                loss_dual_func.backward()
            return loss_dual_func

        for _ in range(2):
            self.dual_func_optimizer.step(closure)
        if np.isnan(np.linalg.norm(self.dual_func.params.detach().numpy())):
            with torch.no_grad():
                self.dual_func.params.copy_(torch.zeros(self.dual_func.shape, dtype=torch.float32))
        return

    def _optimize_dual_func_gd(self, x_tensor, z_tensor, iters):
        for i in range(iters):
            self.dual_func_optimizer.zero_grad()
            _, loss_dual_func = self.objective(x_tensor, z_tensor)
            loss_dual_func.backward()
            self.dual_func_optimizer.step()
            return loss_dual_func

    def _init_training(self, x_tensor, z_tensor, z_val_tensor=None):
        self._set_kernel_z(z_tensor, z_val_tensor)
        self._init_dual_func()
        self._set_optimizers()
        if self.pretrain:
            self._pretrain_theta(x=x_tensor, z=z_tensor)

    def _train_internal(self, x_train, z_train, x_val, z_val, debugging):
        x_tensor = self._to_tensor(x_train)
        x_val_tensor = self._to_tensor(x_val)

        if z_train is not None:
            z_tensor = self._to_tensor(z_train)
            z_val_tensor = self._to_tensor(z_val)
        else:
            z_tensor, z_val_tensor = None, None

        if self.batch_training:
            n = x_train[0].shape[0]
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
                    z_batch = z_tensor[batch_idx] if z_tensor is not None else None
                    obj = self.optimize_step(x_batch, z_batch)
                    # loss.append(obj)
            else:
                obj = self.optimize_step(x_tensor, z_tensor)
                # loss.append(obj)

            if epoch_i % eval_freq_epochs == 0:
                cycle_num += 1
                val_loss = self.calc_validation_metric(x_val, z_val)
                if self.verbose:
                    val_theta_obj, _ = self.objective(x_val_tensor, z_val_tensor)
                    print("epoch %d, theta-obj=%f, val-mmr-loss=%f"
                          % (epoch_i, val_theta_obj.detach().numpy(), val_loss.detach().numpy()))
                mmr.append(float(val_loss))
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
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

    """--------------------- Visualization methods ---------------------"""
    # def compute_weights(self, x, z):
    #     dual_func_psi = self.compute_dual_func_psi(x, z).detach().numpy()
    #     if self.divergence_type == 'log':
    #         weights = 1 / (1 - dual_func_psi) / np.sum(1 / (1 - dual_func_psi))
    #     elif self.divergence_type == 'chi2':
    #         weights = (dual_func_psi + 1) / np.sum(dual_func_psi + 1)
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
    #         self.dual_func.init_params()
    #         self.set_optimizers(self.dual_func)
    #
    #         self.optimize_dual_func(x, z)
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
    #             self.dual_func.init_params()
    #             self.set_optimizers(self.dual_func)
    #
    #             self.optimize_dual_func(x, z)
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
