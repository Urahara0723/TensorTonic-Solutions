import numpy as np
from math import sqrt

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    n = len(x)
    x = np.array(x)
    mean = float(np.mean(x))
    var = float(np.sum((x-mean)**2)/(n-1))
    dev =sqrt(var)
    return {"variance":var,
           "standard_deviation":dev}