import torch
import numpy as np

from fgel.abstract_estimation_method import AbstractEstimationMethod
from fgel.utils.rkhs_utils import get_rbf_kernel


class KernelMMR(AbstractEstimationMethod):
    def __init__(self, model, psi_dim, kernel_args=None, verbose=True):
        AbstractEstimationMethod.__init__(self, model, psi_dim)
        self.kernel_args = kernel_args
        self.kernel_z = None
        self.kernel_z_val = None
        self.verbose = verbose

    def set_kernel(self, z, z_val=None):
        if self.kernel_args is None:
            self.kernel_args = {}
        if self.kernel_z is None:
            self.kernel_z = get_rbf_kernel(z, z, **self.kernel_args)
        if z_val is not None:
            self.kernel_z_val = get_rbf_kernel(z_val, z_val, **self.kernel_args)

    def _fit_internal(self, x, z, x_val, z_val, show_plots=False):
        x_tensor = self._to_tensor(x)
        z_tensor = self._to_tensor(z)
        n_sample = z_tensor.shape[0]

        self.set_kernel(z, z_val)

        optimizer = torch.optim.LBFGS(self.model.parameters(),
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            psi = self.model.psi(x_tensor)
            loss = torch.einsum('ir, ij, jr -> ', psi, self.kernel_z, psi) / (n_sample ** 2)
            loss.backward()
            return loss
        optimizer.step(closure)

        if self.verbose and x_val is not None:
            val_mmr = self._calc_val_mmr(x_val, z_val)
            print("Validation MMR loss: %e" % val_mmr)


if __name__ == '__main__':
    from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment

    theta = 1.7
    noise = 1.0

    exp = HeteroskedasticNoiseExperiment(theta=[theta], noise=noise)
    exp.setup_data(n_train=200, n_val=2000, n_test=20000)

    model = exp.get_model()
    estimator = KernelMMR(model=model, psi_dim=1)

    print('Parameters pre-train: ', estimator.model.get_parameters())
    estimator.fit(exp.x_train, exp.z_train, exp.x_val, exp.z_val)

    train_risk = exp.eval_test_risk(model, exp.x_train)
    test_risk = exp.eval_test_risk(model, exp.x_test)
    print('Parameters: ', np.squeeze(model.get_parameters()), ' True: ', theta)
    print('Train risk: ', train_risk)
    print('Test risk: ', test_risk)