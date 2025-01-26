import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import sigmoid
import softmax

# データ取得関数
def get_data():
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # flatten
  flattened_x_test = np.empty(shape=(x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
  for i in range(len(x_test)):
    flattened_x_test[i] = x_test[i].reshape([x_test[i].shape[0] * x_test[i].shape[1],])

  # normalize
  flattened_x_test = flattened_x_test / 255.0

  return flattened_x_test, y_test

# NNの重みをロードする関数
def init_network():
  with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)

    return network

# NNモデルをつかった予測
def predict(network, x):

  # calculate network
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']
  a1 = np.dot(x, W1) + b1
  Z1 = sigmoid.sigmoid(a1)
  a2 = np.dot(Z1, W2) + b2
  Z2 = sigmoid.sigmoid(a2)
  a3 = np.dot(Z2, W3) + b3
  y = softmax.softmax(a3)

  return y

if __name__=="__main__":
  x, t = get_data()
  network = init_network()
  
  # batch処理
  batch_size = 100

  accuracy_cent = 0
  for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cent += np.sum(p == t[i:i+batch_size])

  print("Accuracy:" + str(float(accuracy_cent / len(x))))