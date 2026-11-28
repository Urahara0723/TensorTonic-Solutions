import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    x = np.array(X, dtype=float)

    mean = np.mean(x, axis=0)
    centered = x - mean

    cov = centered.T @ centered

    std = np.sqrt(np.diag(cov))

    corr = cov / np.outer(std, std)

    return corr