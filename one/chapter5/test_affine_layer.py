import numpy as np
import layer_naive as layer

x = np.array([[0, 0, 0], [1, 2, 3]])
W = np.array([[5, 5, 5], [1, 1, 1], [1, 1, 1]])
b = np.array([1, 2, 3])

affine_layer = layer.Affine(W, b)
out = affine_layer.forward(x)
print("out:", out)

dY = np.array([[1, 2, 3], [4, 5, 6]])
print("dY:", dY)
dx = affine_layer.backward(dY)
print("dx:", dx)
print("dW:", affine_layer.dW)
print("db:", affine_layer.db)
