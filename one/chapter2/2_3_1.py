def AND(x1, x2):
  w1, w2, theta = 0.5, 0.5, 0.7
  tmp = x1*w1 + x2*w2
  if tmp <= theta:
    return 0
  else:
    return 1

if __name__ == "__main__":
  x1_list = [0, 1]
  x2_list = [0, 1]
  for x1 in x1_list:
    for x2 in x2_list:
      print(f'x1 = {x1}, x2 = {x2}, output = {AND(x1, x2)}')