import numpy as np
import pandas as pd
from scipy import special
import lmfit
from functools import wraps

def params_as_dict(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        params = kwargs.get('params', args[1])
        if type(params) is dict:
            return func(*args, **kwargs)
        elif type(params) is pd.Series:
            return func(args[0], params.todict())
        elif type(params) is lmfit.Parameters:
            d = dict(
                zip(params.keys(), map(lambda x: x.value, params.values())))
            return func(args[0], d)
    return wrapper


@params_as_dict
def model(t, params):
    """Model a wire rotated in a power fluid.

    Parameters
    ----------
    t : array-like or scalar time
    params : dict or Parameters object
        containing a, b, K

    Returns
    -------
    time : array-like or scalar time in seconds
    """
    a = params['a']
    b = params['b']
    K = params['K']
    theta = a + b*np.arctan(np.exp(K*t))
    return theta
