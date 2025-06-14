import common.optimizer as opt

def f(x):
    """
    Two dimensional function x**2 / 20 + y**2
    where x[0] is the first dimension and x[1] is the second dimension.

    Parameters:
        x (list or np.ndarray): Input vector with two elements.
    Returns:
        float: The value of the function at the input vector.
    """
    return (x[0] ** 2) / 20 + x[1] ** 2

def main():
    param = {
        "test": 6.45,
    }
    optimizer = opt.Momentum(lr=0.01, momentum=0.9)
    grad = {
        "test": 0.01,
    }

    for i in range(100):
        optimizer.update(param, grad)