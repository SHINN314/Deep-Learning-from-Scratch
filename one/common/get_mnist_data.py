import numpy as np
import tensorflow as tf

# データ取得関数
def get_mnist_data(flatten=True, normalize=True):
  mnist = tf.keras.datasets.mnist
  (x_train, t_train), (x_test, t_test) = mnist.load_data()

  # one-hot encode
  t_train = np.identity(len(t_train))[t_train]
  t_test = np.identity(len(t_test))[t_test]

  # normalize
  if(normalize):
    x_train = x_train / 255.0
    x_test = x_test / 255.0

  # flatten
  if(flatten):
    # initialize flatten numpy array
    flattened_x_train = np.empty(shape=(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    flattened_x_test = np.empty(shape=(x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    # flatten x_train
    for i in range(len(x_train)):
      flattened_x_train[i] = x_train[i].reshape([x_train[i].shape[0] * x_train[i].shape[1],])

    # flatten x_test
    for i in range(len(x_test)):
      flattened_x_test[i] = x_test[i].reshape([x_test[i].shape[0] * x_test[i].shape[1],])

    return flattened_x_train, t_train, flattened_x_test, t_test
  else:
    return x_train, t_train, x_test, t_test
