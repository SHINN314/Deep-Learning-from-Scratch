import sys, os
sys.path.append(os.pardir)
import numpy as np
import common.networks as networks
import common.gradient as gradient

if __name__=="__main__":
  network = networks.simpleNet()
  x = np.array([0.6, 0.9]) # 入力
  p = network.predict(x)
  t = np.array([0, 0, 1]) # 正解ラベル

  def f(W):
    return network.loss(x, t)
    
  dW = gradient.numerical_gradient(f, network.W)
  print(dW)