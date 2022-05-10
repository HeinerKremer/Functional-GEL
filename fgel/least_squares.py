from fgel.abstract_estimation_method import AbstractEstimationMethod
import torch
import numpy as np


class OrdinaryLeastSquares(AbstractEstimationMethod):
    def __init__(self, model, psi_dim):
        AbstractEstimationMethod.__init__(self, model, psi_dim)

    def _fit_internal(self, x, z, x_dev, z_dev, show_plots=False):
        x_tensor = self._to_tensor(x)
        z_tensor = self._to_tensor(z)
        n_sample = z_tensor.shape[0]

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
    from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment

    theta = 1.7
    noise = 1.0

    exp = HeteroskedasticNoiseExperiment(theta=[theta], noise=noise)
    exp.setup_data(n_train=200, n_val=2000, n_test=20000)

    model = exp.get_model()
    estimator = OrdinaryLeastSquares(model=model, psi_dim=1)

    print('Parameters pre-train: ', estimator.model.get_parameters())
    estimator.fit(exp.x_train, exp.z_train, exp.x_val, exp.z_val)

    train_risk = exp.eval_test_risk(model, exp.x_train)
    test_risk = exp.eval_test_risk(model, exp.x_test)
    print('Parameters: ', np.squeeze(model.get_parameters()), ' True: ', theta)
    print('Train risk: ', train_risk)
    print('Test risk: ', test_risk)