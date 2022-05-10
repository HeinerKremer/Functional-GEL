import numpy as np
import torch
import torch.nn as nn


def torch_to_float(tensor):
    return float(tensor.detach().cpu())


def torch_to_np(tensor):
    return tensor.detach().cpu().numpy().astype("float64")


def np_to_tensor(data_array):
    if type(data_array) == list:
        tensor_list = []
        for element in data_array:
            data_tensor = torch.from_numpy(element).float()
            tensor_list.append(data_tensor)
        data_tensor = tensor_list
    else:
        data_tensor = torch.from_numpy(data_array).float()
    return data_tensor


class Parameter(nn.Module):
    def __init__(self, shape=None, n_sample=None):
        super().__init__()
        self.n_sample = n_sample
        self.shape = shape
        self.params = None
        self.init_params()

    def forward(self, data=None):
        return self.params

    def parameters(self, recourse=True):
        return [self.params]

    def init_params(self):
        if self.shape is None:
            assert self.n_sample is not None
            start_val = torch.tensor(1 / self.n_sample * np.ones([self.n_sample, 1]), dtype=torch.float32)
        else:
            start_val = torch.Tensor([1/self.shape[0]]) * torch.ones(self.shape, dtype=torch.float32)
        self.params = torch.nn.Parameter(start_val, requires_grad=True)

    def project_simplex_constraint(self):
        params = self.params.detach().numpy()
        # Set weights to very small values > 0
        params[params <= 0] = 1 / 100 * 1 / self.n_sample
        params = torch.tensor(params / params.sum())
        with torch.no_grad():
            self.params.copy_(params)

    def project_log_input_constraint(self, alpha_rho):
        with torch.no_grad():
            max_val = torch.max(alpha_rho)
            constraint_val = torch.tensor(1 - 1/self.n_sample)
            if max_val > constraint_val:
                # Rescale the length of alpha such that constraint is fulfilled
                alpha = self.params / max_val * constraint_val
                alpha_rho = alpha_rho / max_val * constraint_val
                self.update_params(alpha)
            else:
                alpha = self.params
            return alpha, alpha_rho

    def update_params(self, new_params):
        if not isinstance(new_params, torch.Tensor):
            new_params = torch.Tensor(new_params)
        with torch.no_grad():
            self.params.copy_(new_params.clone().detach())

    def reset_params(self):
        if self.shape is None:
            assert self.n_sample is not None
            start_val = torch.tensor(1 / self.n_sample * np.ones([self.n_sample, 1]), dtype=torch.float32)
        else:
            start_val = torch.Tensor([1/self.shape[0]]) * torch.ones(self.shape, dtype=torch.float32)
        self.update_params(start_val)
