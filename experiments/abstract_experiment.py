


class AbstractExperiment:
    def __init__(self, psi_dim, theta_dim, z_dim):
        self.psi_dim = psi_dim
        self.theta_dim = theta_dim
        self.z_dim = z_dim

    def generate_data(self, num_data):
        raise NotImplementedError

    def get_true_parameters(self):
        raise NotImplementedError