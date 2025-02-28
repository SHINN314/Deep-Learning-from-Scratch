import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import common.networks as networks
import common.get_mnist_data as get_mnist_data

if __name__ == "__main__":
  # get mnist data
  x_train, t_train, x_test, t_test = get_mnist_data.get_mnist_data()

  train_loss_list = []

  # hyper parameters
  iters_num = 1000 # number of learning
  train_size = x_train.shape[0]
  batch_size = 100
  learning_rate = 0.1

  # initialize network
  network = networks.TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

  for i in range(iters_num):
    # get mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # calculate gradient
    grad = network.numerical_gradient(x_batch, t_batch)

    # update parameters
    for key in network.params:
      network.params[key] -= learning_rate * grad[key]

    # save learning process
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

  # visualize learning process
  index_list = [x for x in range(0, iters_num)]
  plt.plot(index_list, train_loss_list)
  plt.xlabel("Iteration")
  plt.ylabel("loss")
  plt.savefig("figure4.png")
