import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.get_mnist_data import get_mnist_data

if __name__=="__main__":
  x_train, t_train, x_test, t_test = get_mnist_data()
  print(x_train.shape)
  print(t_train.shape)