import numpy as np
import functions

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

if __name__=="__main__":
  functions.plot_function_graph(sigmoid)