import numpy as np
import functions

def step_function(x):
  y = x > 0
  return y.astype(int)

if __name__ == "__main__":
  x = np.array([0.1, -2.0, 0.0000001])
  print(f'output step_function {step_function(x)}')
  functions.plot_function_graph(step_function)
