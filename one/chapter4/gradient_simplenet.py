import sys, os
sys.path.append(os.pardir)
import common.networks as networks

if __name__=="__main__":
  network = networks.simpleNet()
  print(network.W)