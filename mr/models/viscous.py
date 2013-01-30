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


def setup_params(angle):
    """The fit parameters' bounds depend on the range of the angle data.
    
    Parameters
    ----------
    angle : an array-like sequence of angle values

    Returns
    -------
    params : Parameters object with correct bounds
    """
    pi = np.pi
    params = lmfit.Parameters()
    params.add('m', value=1.7, min=0)
    params.add('C', value=1.0, min=0)
    params.add('offset', pi/100) # specify bounds below 
    params.add('theta0_plus_offset', pi/50, min=0, max=0.2) # bounding utility 
    params.add('theta0', expr='theta0_plus_offset - offset')

    offset = params['offset']
    angle_min, angle_max = angle.min(), angle.max()
    assert angle_min.size == 1, "argument must be a 1D sequence"
    offset.min, offset.max = -pi/2 - angle_min, pi/2 - angle_max
    return params

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
