import numpy as np
from math import sqrt

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a Python float.
    """
    d = float(np.mean(np.dot(x,x)) + np.mean(np.dot(y,y)) - 2*np.dot(x,y))
    
    return sqrt(d)