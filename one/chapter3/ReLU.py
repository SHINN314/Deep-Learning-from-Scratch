import numpy as np
import functions

def ReLU(x):
  return np.maximum(0, x)

if __name__=="__main__":
  functions.plot_function_graph(ReLU)