import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    N = len(X)
    x = np.array(X)
    mean = np.mean(x,axis=0)
    centered = x - mean
    return np.dot(centered.T, centered)/(N-1)