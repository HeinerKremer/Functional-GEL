


class AbstractExperiment:
    def __init__(self, psi_dim, theta_dim, z_dim):
        self.psi_dim = psi_dim
        self.theta_dim = theta_dim
        self.z_dim = z_dim
        self.x_train, self.z_train = None, None
        self.x_val, self.z_val = None, None
        self.x_test, self.z_test = None, None

    def generate_data(self, num_data):
        raise NotImplementedError

    def setup_data(self, n_train, n_val=None, n_test=None):
        self.x_train, self.z_train = self.generate_data(n_train)
        self.x_val, self.z_val = self.generate_data(n_val)
        self.x_test, self.z_test = self.generate_data(n_test)

    def get_true_parameters(self):
        raise NotImplementedError