import numpy as np

def numerical_gradient(f, x):
  """
  Calculate gradient of each x's element

  Parameters
  ----------
  f: function
    Target of differentiation.

  x: numpy array
    Point to differentiation.

  Returns
  ----------
  grad: numpy array 
    Result of differentiation.
  """
  h = 1e-4
  grad = np.zeros_like(x)

  for idx in range(x.shape[0]):
    # idx成分の微分を計算
    tmp_val = x[idx]
  
    # f(x + h)の計算
    x[idx] = tmp_val + h
    fxh1 = f(x)

    # f(x - h)の計算
    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2.0*h) # 勾配の計算
    x[idx] = tmp_val # 値の初期化

  return grad