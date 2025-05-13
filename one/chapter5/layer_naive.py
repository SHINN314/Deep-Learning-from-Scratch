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