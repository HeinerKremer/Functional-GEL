from fgel.abstract_estimation_method import AbstractEstimationMethod
import torch
import numpy as np


class OrdinaryLeastSquares(AbstractEstimationMethod):
    def __init__(self, model):
        AbstractEstimationMethod.__init__(self, model)

    def _train_internal(self, x, z, x_dev, z_dev, debugging):
        x_tensor = self._to_tensor(x)
        z_tensor = self._to_tensor(z)
        n_sample = x_tensor[0].shape[0]

        optimizer = torch.optim.LBFGS(self.model.parameters(),
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            psi = self.model.psi(x_tensor)
            loss = torch.einsum('ir, ir -> ', psi, psi) / n_sample
            loss.backward()
            return loss
        optimizer.step(closure)


if __name__ == '__main__':
    from experiments.tests import test_mr_estimator, test_cmr_estimator
    test_mr_estimator(estimation_method='OLS', n_runs=2)
    test_cmr_estimator(estimation_method='OLS', n_runs=2)
