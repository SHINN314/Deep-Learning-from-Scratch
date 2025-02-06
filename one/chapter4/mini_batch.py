import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.get_mnist_data import get_mnist_data

if __name__=="__main__":
  x_train, t_train, x_test, t_test = get_mnist_data()
  train_size = x_train.shape[0]
  batch_size = 10
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]
  print(x_batch) # test code
  print(t_batch) # test code