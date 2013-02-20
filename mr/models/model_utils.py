import pandas as pd
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
