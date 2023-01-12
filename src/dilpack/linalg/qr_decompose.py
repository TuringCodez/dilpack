import numpy as np

# Implemention of the Modified Gram-Schimt
def computeQR(matrix):
  m, n = np.shape(matrix)
  # transpose since we are going to work on the columns

  ortho_matrix = matrix.copy()

  for x in range(n):
    pivot_col = np.transpose(ortho_matrix[:,x])
    pivot_col /= np.linalg.norm(pivot_col)
    ortho_matrix[:,x] = np.transpose(pivot_col)
    for current in range(x + 1, n):
      v = np.transpose(ortho_matrix[:,current])
      ortho_matrix[:,current] -= np.transpose((np.dot(pivot_col, v)/np.dot(pivot_col,pivot_col)) * pivot_col)

  R = np.matmul(np.transpose(ortho_matrix), matrix)
  Q = ortho_matrix
 
  return Q, R