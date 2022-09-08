import torch
import torch.nn as nn
import numpy as np
from experiments.abstract_experiment import AbstractExperiment
from fgel.baselines.least_squares import OrdinaryLeastSquares


def generate_data(poisson_param, n_sample):
    y = torch.Tensor(np.random.poisson(lam=poisson_param, size=[n_sample, 1]))
    data = {'t': None, 'y': y, 'z': None}
    return data

class PoissonParameter(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.poisson_parameter = torch.nn.Parameter(torch.tensor([0.5]))

    def forward(self, t=None):
        return self.poisson_parameter


def moment_function(model_evaluation, y):
    mean = torch.mean(y) - model_evaluation
    variance = torch.mean(y**2) - model_evaluation**2 - model_evaluation
    moments = torch.Tensor([mean, variance])
    moments = torch.reshape(moments, (2,1))
    return moments



if __name__ == '__main__':
    n_sample = 100
    poisson_param = 0.5
    train_data = generate_data(poisson_param=poisson_param, n_sample=n_sample)
    validation_data = generate_data(poisson_param=poisson_param, n_sample=n_sample)

    model = PoissonParameter()
    print(model())
    print(moment_function(model(), train_data['y']))
