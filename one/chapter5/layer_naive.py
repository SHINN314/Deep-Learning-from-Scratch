import numpy as np

class MulLayer:
  """
  Multiplication layer

  Attributes
  ----------
  x : numpy.ndarray
      Input data
  y : numpy.ndarray
      Input data
  """
  def __init__(self):
    self.x = None
    self.y = None

  def forward(self, x, y):
    """
    Forward pass
    
    Parameters
    ----------
    x : numpy.ndarray
        Input data
    y : numpy.ndarray
        Input data
    
    Returns
    -------
    numpy.ndarray
        Output data
    """
    self.x = x
    self.y = y
    out = x * y

    return out
  
  def backward(self, dout):
    """
    Backward pass
    
    Parameters
    ----------
    dout : numpy.ndarray
      Gradient of the loss with respect to the output
    
    Returns
    -------
    tuple
      Gradients of the loss with respect to the inputs
    """
    dx = dout * self.y
    dy = dout * self.x

    return dx, dy
  
class AddLayer:
  """
  Addition layer
  
  Attributes
  ----------
  x : numpy.ndarray
      Input data
  y : numpy.ndarray
      Input data
  """

  def __init__(self):
    pass # no need to initialize anything

  def forward(self, x, y):
    """
    Forward pass
    
    Parameters
    ----------
    x : numpy.ndarray
        Input data
    y : numpy.ndarray
        Input data
    
    Returns
    -------
    numpy.ndarray
        Output data
    """
    out = x + y

    return out
  
  def backward(self, dout):
    """
    Backward pass
    
    Parameters
    ----------
    dout : numpy.ndarray
      Gradient of the loss with respect to the output
      
    Returns
    ----------
    tuple
      Gradients of the loss with respect to the inputs
    """

    dx = dout
    dy = dout

    return dx, dy
  
class Relu:
  """
  ReLU layer

  Attributes
  ----------
  mask : numpy.ndarray
      Mask for the input data.
      The elements of numpy.darray are True if the corresponding element of the input data is less than or equal to 0, and False otherwise.
  """

  def __init__(self):
    self.mask = None

  def forward(self, x):
    """
    Forward pass
    
    Parameters
    ----------
    x : numpy.ndarray
        Input data
        
    Returns
    -------
    numpy.ndarray
        Output data
    """
    self.mask = (x <= 0)
    out = x.copy()

    # Set the elements of out to 0 where the corresponding element of the mask is True
    out[self.mask] = 0

    return out
  
  def backward(self, dout):
    """
    Backward pass
    
    Parameters
    ----------
    dout : numpy.ndarray
      Gradient of the loss with respect to the output
    
    Returns
    -------
    numpy.ndarray
      Gradient of the loss with respect to the input
    """
    dout[self.mask] = 0
    dx = dout

    return dx
  
class Sigmoid:
  """
  Sigmoid Layer
  
  Attributes
  ----------
  out : numpy.ndarray
      Output data
  """

  def __init__(self):
    self.out = None

  def forward(self, x):
    """
    Forward pass
    
    Parameters
    ----------
    x : numpy.ndarray
        Input data
        
    Returns
    -------
    numpy.ndarray
        Output data
    """
    out = 1  / (1 + np.exp(-x))
    self.out = out

    return out
  
  def backward(self, dout):
    """
    Backward pass
    
    Parameters
    ----------
    dout : numpy.ndarray
      Gradient of the loss with respect to the output
      
    Returns
    -------
    numpy.ndarray
      Gradient of the loss with respect to the input
    """
    dx = dout * self.out * (1 - self.out)

    return dx
  
class Affine:
  """
  Affine layer
  
  Atrributes
  ----------
  W : numpy.ndarray
      Weight matrix
  b : numpy.ndarray
      Bias vector
  x : numpy.ndarray
      Input data
  dW : numpy.ndarray
      Gradient of the loss with respect to the weight matrix
  db : numpy.ndarray
      Gradient of the loss with respect to the bias vector
  """
  def __init__(self, W, b):
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.db = None

  def forward(self, x):
    """
    Forward pass
    
    Parameters
    ----------
    x : numpy.ndarray
        Input data
    
    Returns
    -------
    numpy.ndarray
        Output data
    """
    self.x = x
    out = np.dot(x, self.W) + self.b

    return out
  
  def backward(self, dout):
    """
    Backward pass
    
    Parameters
    ----------
    dout : numpy.ndarray
      Gradient of the loss with respect to the output
      
    Returns
    -------
    numpy.ndarray
      Gradient of the loss with respect to the input
    """
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis=0) # sum over the batch size

    return dx
    