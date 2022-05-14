import torch
import numpy as np

from fgel.abstract_estimation_method import AbstractEstimationMethod


class KernelMMR(AbstractEstimationMethod):
    def __init__(self, model, kernel_args=None, verbose=False):
        AbstractEstimationMethod.__init__(self, model, kernel_args)
        self.verbose = verbose

    def _fit_internal(self, x, z, x_val, z_val, debugging=False):
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
    from experiments.exp_heteroskedastic import run_heteroskedastic_n_times

    results = run_heteroskedastic_n_times(theta=1.7, noise=1.0, n_train=200, repititions=20,
                                         estimatortype=KernelMMR,)
    print('Thetas: ', results['theta'])