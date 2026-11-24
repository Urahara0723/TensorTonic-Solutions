import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    x = np.array(x)
    
    prob = p**x * (1-p)**(1-x)
    
    mean = float(p)
    var = float(p*(1-p))
    return {"pmf":prob,
            "mean":mean,
            "variance":var
    }