import numpy as np
import matplotlib.pyplot as plt
from diff import numerical_diff

def function_2(x):
  return x[0]**2 + x[1]**2

# ベクトルxにおける勾配を求める関数
def numerical_gradient(f, x):
  h = 1e-4
  grad = np.zeros_like(x)

  for idx in range(x.size):
    tmp_val = x[idx]

    # f(x + h)の計算
    x[idx] = tmp_val + h
    fxh1 = f(x)

    # f(x - h)の計算
    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2.0*h)
    x[idx] = tmp_val # 値の初期化

  return grad

if __name__ == "__main__":
  print(numerical_gradient(function_2, np.array([0.0, 2.0])))