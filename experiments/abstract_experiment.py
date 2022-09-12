
class AbstractExperiment:
    def __init__(self, psi_dim, theta_dim, z_dim):
        self.psi_dim = psi_dim
        self.theta_dim = theta_dim
        self.z_dim = z_dim
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def generate_data(self, num_data):
        raise NotImplementedError

    def prepare_dataset(self, n_train, n_val=None, n_test=None):
        self.train_data = self.generate_data(n_train)
        self.val_data = self.generate_data(n_val)
        self.test_data = self.generate_data(n_test)

    def validation_loss(self, model, val_data):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError

    def get_true_parameters(self):
        raise NotImplementedError