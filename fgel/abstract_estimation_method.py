from fgel.utils.rkhs_utils import get_rbf_kernel, compute_cholesky_factor
from fgel.utils.torch_utils import np_to_tensor
import numpy as np
import torch


class AbstractEstimationMethod:
    def __init__(self, model, kernel_args=None):
        self.model = model
        self.psi_dim = self.model.psi_dim
        self.is_fit = False

        # For validation purposes all methods use the kernel MMR loss and therefore require the kernel Gram matrices
        if kernel_args is None:
            kernel_args = {}
        self.kernel_args = kernel_args
        self.kernel_z = None
        self.k_cholesky = None
        self.kernel_z_val = None

    def fit(self, x, z, x_dev, z_dev, show_plots=False):
        self._fit_internal(x, z, x_dev, z_dev, show_plots=show_plots)
        self.is_fit = True

    def get_trained_parameters(self):
        if not self.is_fit:
            raise RuntimeError("Need to fit model before getting fitted params")
        return self.model.get_parameters()

    def set_kernel(self, z, z_val=None):
        if self.kernel_z is None and z is not None:
            self.kernel_z = get_rbf_kernel(z, z, **self.kernel_args).type(torch.float32)
            self.k_cholesky = torch.tensor(np.transpose(compute_cholesky_factor(self.kernel_z.detach().numpy())))
        if z_val is not None:
            self.kernel_z_val = get_rbf_kernel(z_val, z_val, **self.kernel_args)

    def _calc_val_mmr(self, x_val, z_val):
        if not isinstance(x_val, torch.Tensor):
            x_val = self._to_tensor(x_val)
        if not isinstance(z_val, torch.Tensor):
            z_val = self._to_tensor(z_val)
        n = z_val.shape[0]
        self.set_kernel(z=None, z_val=z_val)
        psi = self.model.psi(x_val)
        loss = torch.einsum('ir, ij, jr -> ', psi, self.kernel_z_val, psi) / (n ** 2)
        return loss

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array)

    def _fit_internal(self, x, z, x_dev, z_dev, show_plots):
        raise NotImplementedError()
