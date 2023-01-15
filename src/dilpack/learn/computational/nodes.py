import numpy as np

class ComputationalGraphNode:
  def __init__(self):
    self.inputs = []

  def gradient(self):
    return []

  def trainable(self):
    return False

  def __repr__(self):
    return f"ComputationalGraphNode"

  def dump(self, indent = 0):
    print(" ".ljust(indent) + f"{self}")
    for idx, input in enumerate(self.input):
      print(" ".ljust(indent) + f"  input {idx+1}:")
      if isinstance(input, ComputationalGraphNode):
        input.dump(indent + 4)
      else:
        print(" ".ljust(indent) + f"  {input}")

class Node(ComputationalGraphNode):
  def __init__(self, A, trainable=False):
    self.A = A
    self.input = [A]
    self.errors = [np.zeros(np.shape(self.A))]
    self.is_trainable = trainable
  
  def shape(self):
    return np.shape(self.A)

  def update(self, A):
    self.A = A
    self.input = [A]
    self.errors = [np.zeros(np.shape(self.A))]

  def forward(self):
    return self.A

  def gradient(self, error):
    return [np.ones_like(self.A) * error]

  def trainable(self):
    return self.is_trainable

  def __repr__(self):
    return f"Value(Trainable={self.is_trainable})"

  

class Add(ComputationalGraphNode):
  def __init__(self, A, B):
    self.input = [A, B]
    self.A = A
    self.B = B
    self.forward_a = np.zeros(self.A.shape())
    self.forward_b = np.zeros(self.B.shape())
    self.errors = [np.zeros(self.A.shape()), np.zeros(self.B.shape())]
  
  def shape(self):
    return np.shape(self.forward_a + self.forward_b)

  def forward(self):
    self.forward_a = self.A.forward()
    self.forward_b = self.B.forward()
    return self.forward_a + self.forward_b

  def gradient(self, error):
    grad_a = np.ones_like(self.forward_a)
    grad_b = np.ones_like(self.forward_b)
    return [grad_a * error, grad_b * error]

  def __repr__(self):
    return f"Add(A,B)"

class Sigmoid(ComputationalGraphNode):
  def __init__(self, A, logistic_slope=2.0):
    self.input = [A]
    self.A = A
    self.logistic_slope = logistic_slope
    self.forward_a = np.zeros(self.A.shape())
    self.errors = [np.zeros(self.A.shape())]

  def shape(self):
    return np.shape(self.forward_a)

  def forward(self):
    self.forward_a = 1.0 / (1.0 + np.exp(-self.logistic_slope * self.A.forward()))
    return self.forward_a

  def gradient(self, error):
    return [self.logistic_slope * self.forward_a * (1. - self.forward_a) * error]

  def __repr__(self):
    return f"SIGMOID(logistic-slope: {self.logistic_slope})"

class Tanh(ComputationalGraphNode):
  def __init__(self, A, alpha = 1.716, beta = 0.667):
    self.input = [A]
    self.A = A
    self.alpha = alpha
    self.beta = beta
    self.errors = [np.zeros(self.A.shape())]
    self.forward_a = np.zeros(self.A.shape())

  def shape(self):
    return np.shape(self.forward_a)

  def forward(self):
    self.forward_a = self.alpha * np.tanh(self.beta * self.A.forward())
    return self.forward_a

  def gradient(self, error):
    return [(self.beta/self.alpha) * (self.alpha - self.forward_a) * (self.alpha + self.forward_a) * error]
  
  def __repr__(self):
    return f"Tanh(A,alpha={alpha},beta={beta})"

class LeakyRelu(ComputationalGraphNode):
  def __init__(self, A, slope =0.01):
    self.input = [A]
    self.input_a = None
    self.A = A
    self.slope = slope
    self.errors = [np.zeros(self.A.shape())]
    self.forward_a = np.zeros(self.A.shape())

  def shape(self):
    return np.shape(self.forward_a)

  def forward(self):
    self.input_a = self.A.forward()

    output = (self.input_a > 0).astype(self.input_a.dtype) * self.input_a
    leaky_output = (self.input_a < 0).astype(self.input_a.dtype) * self.input_a * self.slope
    self.forward_a = output + leaky_output
    return self.forward_a

  def gradient(self, error):
    input = self.input_a
    output = (input > 0).astype(input.dtype)
    leaky_output = (input < 0).astype(input.dtype) * self.slope
    return [(output + leaky_output) * error]

  def __repr__(self):
    return f"LeakyRelu(A,slope={self.slope})"

class MatMul(ComputationalGraphNode):
  def __init__(self, A,B):
    self.input = [A, B]
    self.A = A
    self.B = B
    self.errors = [np.zeros(self.A.shape()),np.zeros(self.B.shape())]
    self.forward_a = np.zeros(self.A.shape())
    self.forward_b = np.zeros(self.B.shape())

  def shape(self):
    m1, _ = self.A.shape()
    _, n2 = self.B.shape()
    return [m1, n2]

  def forward(self):
    self.forward_a = self.A.forward()
    self.forward_b = self.B.forward()
    return np.matmul(self.forward_a, self.forward_b)

  def gradient(self, error):
    grad_a = np.matmul(error, np.transpose(self.forward_b))
    grad_b = np.transpose(np.matmul(np.transpose(error), self.forward_a))
    assert np.shape(grad_a) == self.A.shape()
    assert np.shape(grad_b) == self.B.shape()
    return [grad_a, grad_b]

  def __repr__(self):
    return f"MatMul(A,B)"

