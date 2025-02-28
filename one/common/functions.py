import numpy as np

def sigmoid(x, a=1.0):
  """
  Calculate sigmoid function.

  Parameters
  ----------
  x: numpy array
    Array of linear sum values.

  a: float, default 1.0
    Gain.

  Returns
  ----------
  output: float
    Output of sigmoid function
  """
  output = 1 / (1 + np.exp(-a * x))
  return output

def softmax(x):
  """
  Calculate each element's array.

  Parameters
  ----------
  x: numpy array
    input array from hidden layer.

  Returns
  ----------
  probability: numpy array
    calculated each element's probability array.
  """
  max_num = np.max(x)
  probability = np.exp(x - max_num) / np.sum(np.exp(x - max_num))
  
  return probability

def cross_entropy_error(y, t):
  """
  Clculate categorical variable's error.

  Parameters
  ----------
  y: numpy array
    NN's output.

  t: numpy array
    teacher data.

  Returns
  ----------
  error: float
    calculated categorical error between y and t.
  """
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batch_size = y.shape[0]
  error = -np.sum(t * np.log(y + 1e-7)) / batch_size
  return error