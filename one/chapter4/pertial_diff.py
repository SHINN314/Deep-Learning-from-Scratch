import numpy as np
import matplotlib.pyplot as plt
from diff import numerical_diff

def function_2(x):
  return x[0]**2 + x[1]**2

if __name__ == "__main__":
  x = np.empty([2, np.arange(-3, 3, 0.1).shape[0]])
  x[0] = np.arange(-3, 3, 0.1)
  x[1] = np.arange(-3, 3, 0.1)
  y = function_2(x)
  y = y.reshape([1, y.shape[0]])

  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")
  ax.plot_surface(x[0], x[1], y)
  plt.savefig("figure2.png")
 