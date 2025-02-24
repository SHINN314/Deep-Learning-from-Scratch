import numpy as np
import common.functions as functions

class simpleNet:
  """
  One layer network.
  The network is composed of Dence layer and softmax layer.
  
  Attributes
  ----------
  W: numpy array
    NN's weights.
  """
  def __init__(self):
    self.W = np.random.randn(2, 3) # Initialize weight matrix using Gaussian distribution

  def predict(self, x):
    """
    Predict values for inputs based on learning results.

    Parameters
    ----------
    x: numpy array(2 dimension)
      Input data.
    """
    pred = np.dot(x, self.W)
    return pred
  
  def loss(self, x, t):
    """
    Calculate error between predict and true values using cross entropy error function.

    Parameters
    ----------
    x: numpy array
      Input data.

    t: numpy array
      Teacher data.
    """
    z = self.predict(x)
    y = functions.softmax(z)
    loss = functions.cross_entropy_error(y, t)

    return loss

if __name__=="__main__":
  # test code
  network = simpleNet()
  print(network.W)