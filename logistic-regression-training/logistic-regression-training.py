import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 500) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for i in range(steps):
      z = X@w + b
      y_pred = _sigmoid(z)
      eps = 1e-15
      Loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log ((1 - y_pred) + eps))
      dw = X.T@(y_pred - y)/n_samples
      db = np.mean(y_pred - y)
      w -= lr*dw
      b -= lr*db
    
    return w,b

