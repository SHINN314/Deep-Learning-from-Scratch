import numpy as np

class SDG:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Attributes:
        lr (float): Learning rate for the optimizer.
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        """
        Update parameters using the gradients.

        Parameters:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients corresponding to the parameters.
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    """
    Momentum optimizer.

    Attributes:
        lr (float): Learning rate for the optimizer.
        momentum (float): Momentum factor.
        v (dict): Dictionary to store the velocity for each parameter.
    """
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        """
        Update parameters using the gradients with momentum.

        Parameters:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients corresponding to the parameters.
        """
        if self.v is None:
            self.v = {}

            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]
