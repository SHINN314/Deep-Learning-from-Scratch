import numpy as np

def softmax(x):
  c = np.max(x)
  exp_x = np.exp(x - c) # オーバーフロー対策
  sum_exp_x = np.sum(exp_x)
  return exp_x / sum_exp_x

def softmax_origin(x):
  exp_x = np.exp(x)
  sum_exp_x = np.sum(exp_x)
  return exp_x / sum_exp_x

if __name__=="__main__":
  a = np.array([0.3, 2.9, 4.0])
  print("softmax(a) = ", softmax(a))
  print("softmax_origin = ", softmax_origin(a))

  b = np.array([1010, 1000, 990])
  print("softmax = ", softmax(b))
  print("softmax_origin = ", softmax_origin(b))