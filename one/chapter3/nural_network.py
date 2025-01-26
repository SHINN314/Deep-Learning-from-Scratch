import numpy as np
import sigmoid
import identity

def linear_combination(x, W):
  if x.shape[0] != W.shape[0]:
    print("次元があっていません。やり直してください。")
    return
  elif x.ndim >= 2:
    print("入力ベクトルは第1引数に入れてください")
    return

  return np.dot(x, W)

def NN_layer(x, W, B, activate):
  if x.shape[0] != W.shape[0]:
    print("次元があっていません。やり直してください。")
    return
  elif x.ndim >= 2:
    print("入力ベクトルは第1引数に入れてください")
    return

  return activate(np.dot(x, W) + B)

if __name__=="__main__":
  # 第1層の実装
  x = np.array([1.0, 0.5])
  W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  B1 = np.array([0.1, 0.2, 0.3])
  Z1 = NN_layer(x, W1, B1, sigmoid.sigmoid)
  print("第1層の出力", Z1)

  # 第2層の実装
  W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  B2 = np.array([0.1, 0.2])
  Z2 = NN_layer(Z1, W2, B2, sigmoid.sigmoid)
  print("第2層の出力", Z2)

  # 第3層の出力(出力層)
  W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
  B3 = np.array([0.1, 0.2])
  A3 = NN_layer(Z2, W3, B3, identity.identity_function)
  print("第3層の出力", A3)
