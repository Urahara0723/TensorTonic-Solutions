import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    A = np.array(A)
    rows, columns = A.shape
    A_T = np.zeros((columns,rows))

    for i in range(rows):
       for j in range(columns):
           A_T[j,i] = A[i,j]

    return A_T
