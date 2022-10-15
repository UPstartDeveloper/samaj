import numpy as np


def sigmoid(x: np.array) -> np.array:
    """
    Given an array of activation values, we return 
    an array of probabilities between 0-1. 

    They will NOT necessarily add up to 1.

    This is mainly intended for classification problems.
    """
    return 1.0 / (1.0 + np.exp(-x))