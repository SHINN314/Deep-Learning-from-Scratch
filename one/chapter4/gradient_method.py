import numpy as np
import matplotlib.pyplot as plt
from pertial_diff import numerical_gradient
from pertial_diff import function_2

"""
f: 最適化したい関数
init_x: 初期値
lr: 学習率
step_num: 更新回数
"""
def gradient_descent(f, init_x, lr=0.1, step_num=100):
  x = init_x

  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x = x - lr * grad

    # グラフにプロット
    plt.plot(x[0], x[1], marker=".", color="blue")
    
  plt.xlim(-4, 4)
  plt.ylim(-4, 4)
  plt.title("gradient method steps")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.savefig("figure3.png")

  return x

if __name__=="__main__":
  init_x = np.array([-3.0, 4.0])
  print(gradient_descent(function_2, init_x=init_x))
