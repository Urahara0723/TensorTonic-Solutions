import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:

    matrix = np.array(matrix)

    if axis is None:
        if norm_type == "l1":
            norm = np.sum(np.abs(matrix))
        elif norm_type == "l2":
            norm = np.linalg.norm(matrix, ord="fro")
        else:
            norm = np.max(np.abs(matrix))

    else:
        if norm_type == "l1":
            norm = np.linalg.norm(matrix, ord=1, axis=axis, keepdims=True)
        elif norm_type == "l2":
            norm = np.linalg.norm(matrix, ord=2, axis=axis, keepdims=True)
        else:
            norm = np.max(np.abs(matrix), axis=axis, keepdims=True)


    norm = np.where(norm == 0, 1, norm)
    return matrix / norm