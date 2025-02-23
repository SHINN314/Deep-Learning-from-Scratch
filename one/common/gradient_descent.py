import numpy as np
import matplotlib.pyplot as plt
from gradient import numerical_gradient

def gradient_descent(f, init_x, lr=0.1, step_num=100):
  """
  Calculate function f's minimum, local minimum value, or saddle point

  Parameters
  ----------
  f: function
    Optimizetion function
  
  init_x: numpy array
    Initial values

  lr: float (default: 0.1)
    Learning rate

  step_num: int (default: 100)
    Number of times updating minimum values is repeated"

  Returns
  ----------
  x: numpy array
    f's minimum, local minimum value or saddle point
  """
  x = init_x

  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x = x - lr * grad

  return x

