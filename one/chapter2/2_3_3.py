import numpy as np
import test_case

def AND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.7
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

def OR(x1, x2):
  x = np.array([x1, x2])
  w = np.array([1, 1])
  b = -1
  tmp = np.sum(w*x) + b
  if tmp >= 0:
    return 1
  else:
    return 0

def NOT(x):
  if x >= 1:
    return 0
  else:
    return 1

def XOR(x1, x2):
  nand_output = NOT(AND(x1, x2))
  or_output = OR(x1, x2)
  return AND(nand_output, or_output)

if __name__=="__main__":
  test_case.test_case(XOR)