import ..common.optimizer as opt
import common.gradient as grad

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
    # Initialize parameters and optimizer
    param = {
        "test": [-7.0, 2.0],
    }
    grad = {
        "test": grad.numerical_gradient(f, param["test"]),
    }
    optimizer = opt.Momentum(lr=0.01, momentum=0.9)

    for i in range(100):
        optimizer.update(param, grad)

        # update parameter and gradient
        param["test"] += optimizer.v["test"]
        grad["test"] = grad.numerical_gradient(f, param["test"])


    print("Final parameters:", param["test"])