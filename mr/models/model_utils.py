import pandas as pd
import lmfit
from functools import wraps
import numpy as np

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

def _going_up(a, threshold=0.1, slope_threshold=0.001):
    slope = pd.rolling_mean(a, 100, center=True).diff()
    crossing = ((a > threshold) & (a.shift(1) < threshold) & (slope > slope_threshold)) # boolean
    return crossing

def _going_down(a, threshold = np.pi/2 - 0.1, slope_threshold=-0.001):
    slope = pd.rolling_mean(a, 100, center=True).diff()
    crossing = ((a < threshold) & (a.shift(1) > threshold) & (slope < slope_threshold)) # boolean
    return crossing

def group_curve(a):
    crossing = _going_up(a) | _going_down(a)
    counter = crossing.astype(np.int).cumsum().shift(-2)
    counter.fillna(method='ffill', inplace=True) # trailing NaNs
    return a.groupby(counter)

def _flip_columns(df):
    upsidedown = df.irow(0) > np.pi/4
    df[df.columns[upsidedown]] *= -1
    df[df.columns[upsidedown]] += np.pi/2
    return df

def split_curve(group, B):
    d = dict()
    for k, v in B.iterkv():
        d[v] = group.get_group(k).reset_index(drop=True)
    s = pd.concat(d, axis=1)
    _flip_columns(s)
    return s
