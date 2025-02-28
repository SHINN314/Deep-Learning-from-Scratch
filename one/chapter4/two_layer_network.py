import sys, os
sys.path.append(os.pardir)
import numpy as np
import common.networks as networks

if __name__=="__main__":
  net = networks.TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
  print(net)
  for param in net.params.values():
    print(param.shape)

  x = np.random.rand(100, 784) # dummy data
  t = np.random.rand(100, 10) # dummy teacher data

  grads = net.numerical_gradient(x, t)

  # output gradient data in grads
  for gradient in grads.values():
    print(gradient.shape)