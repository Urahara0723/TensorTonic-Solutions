import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    # Ensure (n, d)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    
    # Feature check
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test must have same number of features")
    
    n_train = X_train.shape[0]
    
    # Compute distances
    diff = X_test[:, None, :] - X_train[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Get sorted indices
    sorted_idx = np.argsort(dist_sq, axis=1)
    
    # Take available neighbors
    k_actual = min(k, n_train)
    indices = sorted_idx[:, :k_actual]
    
    # Pad with -1 if needed
    if k > n_train:
        pad_width = k - n_train
        pad = -1 * np.ones((indices.shape[0], pad_width), dtype=int)
        indices = np.hstack((indices, pad))
    
    return indices