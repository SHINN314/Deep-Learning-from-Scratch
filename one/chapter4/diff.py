import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
  h = 1e-4
  return (f(x + h) - f(x - h)) / (2*h)

def function_1(x):
  return 0.01 * x ** 2 + 0.1 * x

if __name__ == "__main__":
  # 微分の結果をプロット
  x = np.arange(0.0, 20.0, 0.1)
  y = function_1(x)
  print(numerical_diff(function_1, 5.0))
  y_prime = numerical_diff(function_1, 5.0) * (x - 5.0) + function_1(5.0)
  plt.xlabel("X")
  plt.ylabel("f(x)")
  plt.xlim(0, 20.0)
  plt.plot(x, y)
  plt.plot(x, y_prime)
  plt.savefig("figure1.png")