import torch
import torch.nn as nn
import numpy as np
from experiments.abstract_experiment import AbstractExperiment
from fgel.least_squares import OrdinaryLeastSquares


def eval_model(t, theta, numpy=False):
    if not numpy:
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        return torch.sum(t * theta.reshape(1, -1), dim=1, keepdim=True).float() ** 1
    else:
        return np.sum(t * theta.reshape(1, -1), axis=1, keepdims=True) ** 1


class Model(nn.Module):
    def __init__(self, dim_theta):
        nn.Module.__init__(self)
        self.theta = nn.Parameter(torch.FloatTensor([[0.5] * dim_theta]))

    def forward(self, t):
        return eval_model(t, torch.reshape(self.theta, [1, -1]))

    def psi(self, data):
        t, y = torch.Tensor(data[0]), torch.Tensor(data[1])
        return self.forward(t) - y

    def get_parameters(self):
        param_tensor = self.theta.data
        return param_tensor.detach().numpy()

    def initialize(self):
        nn.init.normal_(self.theta)


class HeteroskedasticNoiseExperiment(AbstractExperiment):
    def __init__(self, theta, noise=1.0, heteroskedastic=False):
        self.theta0 = np.asarray(theta).reshape(1, -1)
        self.dim_theta = np.shape(self.theta0)[1]
        self.noise = noise
        self.heteroskedastic = heteroskedastic
        super().__init__(self, theta_dim=self.dim_theta, z_dim=self.dim_theta)

    def get_model(self):
        return Model(dim_theta=self.dim_theta)

    def generate_data(self, num_data):
        if num_data is None:
            return None, None
        t = np.exp(np.random.uniform(-1.5, 1.5, (num_data, self.dim_theta)))
        error1 = []
        if self.heteroskedastic:
            for i in range(num_data):
                error1.append(np.random.normal(0, self.noise * np.abs(t[i, 0]) ** 2))
            error1 = np.asarray(error1).reshape([num_data, 1])
        else:
            error1 = np.random.normal(0, self.noise, [num_data, 1])
        y = eval_model(t, self.theta0, numpy=True) + error1
        x = [t, y]
        z = t
        return x, z

    def get_true_parameters(self):
        return np.array(self.theta0)

    def eval_test_risk(self, model, data_test):
        t_test = data_test[0].reshape(-1, 1)
        y_test = eval_model(t_test, self.theta0, numpy=True)
        y_pred = model.forward(torch.tensor(data_test[0])).detach().numpy()
        return float(((y_test - y_pred) ** 2).mean())


if __name__ == '__main__':
    theta = 1.7
    noise = 1.0

    exp = HeteroskedasticNoiseExperiment(theta=[theta], noise=noise)
    exp.setup_data(n_train=20, n_val=200, n_test=20000)

    model = exp.get_model()
    estimator = OrdinaryLeastSquares(model=model, psi_dim=1)

    print('Parameters pre-train: ', estimator.model.get_parameters())
    estimator.fit(exp.x_train, exp.z_train, exp.x_val, exp.z_val)

    train_risk = exp.eval_test_risk(model, exp.x_train)
    test_risk = exp.eval_test_risk(model, exp.x_test)
    print('Parameters: ', np.squeeze(model.get_parameters()), ' True: ', theta)
    print('Train risk: ', train_risk)
    print('Test risk: ', test_risk)