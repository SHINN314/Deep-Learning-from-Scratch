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