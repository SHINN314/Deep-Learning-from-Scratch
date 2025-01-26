import numpy as np
import matplotlib.pyplot as plt

def plot_function_graph(func):
  x = np.arange(-5.0, 5.0, 0.1)
  y = func(x)
  plt.plot(x, y)
  plt.ylim(-0.1, 1.1)
  plt.show()