import numpy as np
import matplotlib.pyplot as plt
from diff import numerical_diff

def function_2(x, y):
  return x**2 + y**2

if __name__ == "__main__":
  # set data
  x = np.arange(-3, 3, 0.1)
  y = np.arange(-3, 3, 0.1)

  # 格子点の情報をnumpy arrayに保存
  X, Y = np.meshgrid(x, y)
  Z = function_2(X, Y)

  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")
  ax.plot_surface(X, Y, Z)
  ax.set_aspect("auto")
  plt.savefig("figure2.png")
 