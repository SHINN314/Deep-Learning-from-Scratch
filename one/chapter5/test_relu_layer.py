import numpy as np
import layer_naive as layer

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

relu_layer = layer.Relu()
out = relu_layer.forward(x)
print(out)
print(relu_layer.mask)

dout = np.array([[1.0, 0.5], [2.0, 3.0]])
dx = relu_layer.backward(dout)
print(dx)