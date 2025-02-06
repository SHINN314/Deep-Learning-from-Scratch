import numpy as np

def cross_entropy_error(y, t):
  delta = 1e-7 # log(0)防止のため
  if y.ndim == 1:
    # batch_size == 1のときに配列の形を変える
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batch_size = y.shape[0]
  return -np.sum(t * np.log(y + delta)) / batch_size

if __name__=="__main__":
  t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
  y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
  print("error=", cross_entropy_error(y, t))

  y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
  print("error=", cross_entropy_error(y, t))