def test_case(func):
  x1_list = [0, 1]
  x2_list = [0, 1]
  for x1 in x1_list:
    for x2 in x2_list:
      print(f'x1 = {x1}, x2 = {x2}, output = {func(x1, x2)}')