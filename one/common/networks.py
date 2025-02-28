import numpy as np
import common.functions as functions
import common.gradient as gradient

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
  
class TwoLayerNet:
  """
  Tow layers network model.

  Attribute
  ----------
  Params: dict
    Store weights and biases with layer's name.
  """

  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    """
    Parameters
    -----------
    input_size: int
      Input data's size.

    hidden_size: int
      Number of hidden layer's nodes.

    output_size: int
      Output data's size.

    weight_init_std:
      Init weights standard value.
    """
    # initialize weights
    self.params = {}
    self.params["W1"] = weight_init_std * np.random.rand(input_size, hidden_size) # hidden layer's weights
    self.params["b1"] = np.zeros(hidden_size) # hidden layer's bias
    self.params["W2"] = weight_init_std * np.random.rand(hidden_size, output_size) # output layer's weights
    self.params["b2"] = np.zeros(output_size) # output layer's bias

  def predict(self, x):
    """
    Predict from input data.

    Parameters
    ----------
    x: numpy array
      Input data.

    Returns
    ----------
    y: numpy array
      Predicted data.
    """
    W1, W2 = self.params["W1"], self.params["W2"]
    b1, b2 = self.params["b1"], self.params["b2"]

    # hidden layer
    a1 = np.dot(x, W1) + b1
    z1 = functions.sigmoid(a1)
    # output layer
    a2 = np.dot(z1, W2) + b2
    y = functions.softmax(a2)

    return y

  def loss(self, x, t):
    """
    Calculate fross entropy error between predict data and teacher data.

    Parameters
    x: numpy array
      Input data.

    t: numpy array
      Teacher data.

    Returns
    ----------
    error: float
      Cross entropy error.
    """
    y = self.predict(x)
    error = functions.cross_entropy_error(y, t)
    return error
  
  def accuracy(self, x, t):
    """
    Calculate predict accuracy

    Parameters
    ----------
    x: numpy array
      Input data.

    t: numpy array
      Teacher data.

    Returns
    ----------
    accuracy: float
      Correcte output rate.
    """
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy
  
  def numerical_gradient(self, x, t):
    """
    Calculate gradient at point x.

    Parameters
    ----------
    x: numpy array
      Input data.

    t: numpy array
      Teacher data.

    Returns
    ----------
    grads: dict
      Dictionary which have each layer's gradient.
    """
    loss_W = lambda W: self.loss(x, t)

    grads = {}

    grads["W1"] = gradient.numerical_gradient(loss_W, self.params["W1"])
    grads["b1"] = gradient.numerical_gradient(loss_W, self.params["b1"])
    grads["W2"] = gradient.numerical_gradient(loss_W, self.params["W2"])
    grads["b2"] = gradient.numerical_gradient(loss_W, self.params["b2"])

    return grads