import dilpack.learn.computational.nodes as nodes

# Recursive backpropagation algorithm
def backpropagate(node, error = 0.0):
  if (isinstance(node, nodes.ComputationalGraphNode)):
    gradients_per_input = node.gradient(error)
    for idx, grad in enumerate(gradients_per_input):
      node.errors[idx] = grad
      backpropagate(node.input[idx], node.errors[idx])

def update(node, learning_rate = 0.01):
  if isinstance(node, nodes.ComputationalGraphNode):
    for idx, input in enumerate(node.input):
      if node.trainable():
        input += node.errors[idx] * learning_rate
      update(input, learning_rate)