import torch
import torch.nn as nn
import numpy as np
from experiments.abstract_experiment import AbstractExperiment


class PoissonParameter(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.poisson_parameter = torch.nn.Parameter(torch.tensor([[0.5]]))

    def forward(self, t=None):
        return self.poisson_parameter


class PoissonExperiment(AbstractExperiment):
    def __init__(self, poisson_param):
        super().__init__(dim_psi=2, dim_theta=1, dim_z=None)
        self.poisson_param = poisson_param

    def init_model(self):
        return PoissonParameter()

    def generate_data(self, n_sample):
        y = torch.Tensor(np.random.poisson(lam=self.poisson_param, size=[n_sample, 1]))
        data = {'t': y, 'y': y, 'z': None}
        return data

    @staticmethod
    def moment_function(model_evaluation, y):
        mean = torch.Tensor(y - model_evaluation)
        variance = torch.Tensor((y - model_evaluation)**2)
        moments = torch.cat([mean, variance], dim=1)
        return moments


if __name__ == '__main__':
    from fgel.estimation import estimation

    np.random.seed(12345)
    torch.random.manual_seed(12345)

    exp = PoissonExperiment(poisson_param=5)
    exp.prepare_dataset(n_train=10, n_val=100, n_test=1000)
    model = exp.init_model()

    print(np.mean(exp.moment_function(model(), exp.train_data['y']).detach().numpy(), axis=0))

    trained_model, stats = estimation(model=model,
                                      train_data=exp.train_data,
                                      moment_function=exp.moment_function,
                                      estimation_method='GEL',
                                      estimator_kwargs=None, hyperparams=None,
                                      validation_data=exp.val_data, val_loss_func=exp.validation_loss,
                                      verbose=True
                                      )

    print(f'True param: {exp.poisson_param} \n'
          f'Param estimate: {np.squeeze(trained_model.get_parameters())} \n'
          f'MSE: {np.mean(np.square(np.squeeze(trained_model.get_parameters()) - exp.poisson_param))}\n'
          f'Moment function: {np.squeeze(np.mean(exp.moment_function(trained_model(exp.train_data["t"]), exp.train_data["y"]).detach().numpy(), axis=0))}')

