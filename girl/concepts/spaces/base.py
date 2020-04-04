""" Directly used from [rlpyt](https://github.com/astooke/rlpyt)
"""
import numpy as np
from exptools.collections import namedarraytuple

class Space():
    """
    Common definitions for observations and actions.
    """
    def __init__(self):
        """ Specifying what attribute the instance need to have, but don't call this method.
        """
        self.high = 0
        self.low = 0
        self.shape = tuple()
        raise RuntimeError("Control flow should not reach here, you should not call this")
        
    def contains(self, x):
        """
        Telling whether the input array is in the boudary if this space.
        """
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

    def clamp(self, x):
        """
        Clamp all elements into the boundary of this Space.
        And the elements of `x` will be changed
        """
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        x[x > self.high] = self.high
        x[x < self.low] = self.low
        return x

    def sample(self):
        """
        Uniformly randomly sample a random element of this space.
        """
        raise NotImplementedError

    def null_value(self):
        """
        Return a null value used to fill for absence of element.
        """
        raise NotImplementedError
