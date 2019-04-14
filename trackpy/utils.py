from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import logging
import collections
import functools
import re
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from distutils.version import LooseVersion

import pandas as pd
import numpy as np
import scipy
from scipy import stats
import yaml

import trackpy

# Set is_pandas_since_016 for use elsewhere.
# Pandas >= 0.16.0 lets us check if a DataFrame is a view.
try:
    is_pandas_since_016 = (LooseVersion(pd.__version__) >=
                           LooseVersion('0.16.0'))
except ValueError:  # Probably a development version
    is_pandas_since_016 = True

try:
    is_pandas_since_017 = (LooseVersion(pd.__version__) >=
                           LooseVersion('0.17.0'))
except ValueError:  # Probably a development version
    is_pandas_since_017 = True

try:
    is_pandas_since_018 = (LooseVersion(pd.__version__) >=
                           LooseVersion('0.18.0'))
except ValueError:  # Probably a development version
    is_pandas_since_018 = True
try:
    is_pandas_since_023 = (LooseVersion(pd.__version__) >=
                           LooseVersion('0.23.0'))
except ValueError:  # Probably a development version
    is_pandas_since_023 = True

# Wrap the scipy cKDTree to work around a bug in scipy 0.18.0
try:
    is_scipy_018 = LooseVersion(scipy.__version__) == LooseVersion('0.18.0')
except ValueError:  # Probably a development version
    is_scipy_018 = False


if is_scipy_018:
    from scipy.spatial import KDTree as cKDTree
    warnings.warn("Due to a bug in Scipy 0.18.0, the (faster) cKDTree cannot "
                  "be used. For better linking performance, upgrade or "
                  "downgrade scipy.")
else:
    from scipy.spatial import cKDTree

try:
    is_scipy_since_100 = LooseVersion(scipy.__version__) >= LooseVersion('1.0.0')
except ValueError:  # Probably a development version
    is_scipy_since_100 = True


def fit_powerlaw(data, plot=True, **kwargs):
    """Fit a powerlaw by doing a linear regression in log space."""
    ys = pd.DataFrame(data)
    x = pd.Series(data.index.values, index=data.index, dtype=np.float64)
    values = pd.DataFrame(index=['n', 'A'])
    fits = {}
    for col in ys:
        y = ys[col].dropna()
        slope, intercept, r, p, stderr = \
            stats.linregress(np.log(x), np.log(y))
        values[col] = [slope, np.exp(intercept)]
        fits[col] = x.apply(lambda x: np.exp(intercept)*x**slope)
    values = values.T
    fits = pandas_concat(fits, axis=1)
    if plot:
        from trackpy import plots
        plots.fit(data, fits, logx=True, logy=True, legend=False, **kwargs)
    return values


class memo(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    http://wiki.python.org/moin/PythonDecoratorLibrary#Memoize """
    def __init__(self, func):
        self.func = func
        self.cache = {}
        functools.update_wrapper(self, func)

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            warnings.warn("A memoization cache is being used on an uncacheable " +
                          "object. Proceeding by bypassing the cache.",
                          UserWarning)
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value
# This code trips up numba. It's nice for development
# but it shouldn't matter for users.
#   def __repr__(self):
#      '''Return the function's docstring.'''
#      return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


def extract(pattern, string, group, convert=None):
    """Extract a pattern from a string. Optionally, convert it
    to a desired type (float, timestamp, etc.) by specifying a function.
    When the pattern is not found, gracefully return None."""
    # group may be 1, (1,) or (1, 2).
    if type(group) is int:
        grp = (group,)
    elif type(group) is tuple:
        grp = group
    assert type(grp) is tuple, "The arg 'group' should be an int or a tuple."
    try:
        result = re.search(pattern, string, re.DOTALL).group(*grp)
    except AttributeError:
        # For easy unpacking, when a tuple is expected, return a tuple of Nones.
        return None if type(group) is int else (None,)*len(group)
    return convert(result) if convert else result


def timestamp(ts_string):
    "Convert a timestamp string to a datetime type."
    if ts_string is None:
        return None
    return datetime.strptime(ts_string, '%Y-%m-%d %H:%M:%S')


def time_interval(raw):
    "Convert a time interval string into a timedelta type."
    if raw is None:
        return None
    m = re.match('([0-9][0-9]):([0-5][0-9]):([0-5][0-9])', raw)
    h, m, s = map(int, m.group(1, 2, 3))
    return timedelta(hours=h, minutes=m, seconds=s)


def suppress_plotting():
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg') # does not plot to screen


# HH:MM:SS, H:MM:SS, MM:SS, M:SS all OK
lazy_timestamp_pat = r'\d?\d?:?\d?\d:\d\d'

# a time stamp followed by any text comment
ltp = lazy_timestamp_pat
video_log_pattern = r'(' + ltp + r')-?(' + ltp + r')? ?(RF)?(.+)?'


def lazy_timestamp(partial_timestamp):
    """Regularize a lazy timestamp like '0:37' -> '00:00:37'.
HH:MM:SS, H:MM:SS, MM:SS, and M:SS all OK.

    Parameters
    ----------
    partial_timestamp : string or other object

    Returns
    -------
    regularized string
    """
    if not isinstance(partial_timestamp, str):
        # might be NaN or other unprocessable entry
        return partial_timestamp
    input_format = '\d?\d?:?\d?\d:\d\d'
    if not re.match(input_format, partial_timestamp):
        raise ValueError("Input string cannot be regularized.")
    partial_digits = list(partial_timestamp)
    digits = ['0', '0', ':', '0', '0', ':', '0', '0']
    digits[-len(partial_digits):] = partial_digits
    return ''.join(digits)


def timedelta_to_frame(timedeltas, fps):
    """Convert timedelta times into frame numbers.

    Parameters
    ----------
    timedelta : DataFrame or Series of timedelta64 datatype
    fps : frames per second (integer)

    Result
    ------
    DataFrame

    Note
    ----
    This sounds like a stupidly easy operation, but handling missing data
    and multiplication is tricky with timedeltas.
    """
    ns = timedeltas.values
    seconds = ns * 1e-9
    frame_numbers = seconds*fps
    result = pd.DataFrame(frame_numbers, dtype=np.int64,
                          index=timedeltas.index, columns=timedeltas.columns)
    result = result.where(timedeltas.notnull(), np.nan)
    return result


def random_walk(N):
    return np.cumsum(np.random.randn(N), 1)


def record_meta(meta_data, file_obj):
    file_obj.write(yaml.dump(meta_data, default_flow_style=False))

def validate_tuple(value, ndim):
    if not hasattr(value, '__iter__'):
        return (value,) * ndim
    if len(value) == ndim:
        return tuple(value)
    raise ValueError("List length should have same length as image dimensions.")


try:
    from IPython.core.display import clear_output
except ImportError:
    pass


def make_pandas_strict():
    """Configure Pandas to raise an exception for "chained assignments."

    This is useful during tests.
    See http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

    Does nothing for Pandas versions before 0.13.0.
    """
    if LooseVersion(pd.__version__) >= LooseVersion('0.13.0'):
        pd.set_option('mode.chained_assignment', 'raise')


class IPythonStreamHandler(logging.StreamHandler):
    "A StreamHandler for logging that clears output between entries."
    def emit(self, s):
        clear_output(wait=True)
        print(s.getMessage())
    def flush(self):
        sys.stdout.flush()


FORMAT = "%(name)s.%(funcName)s:  %(message)s"
formatter = logging.Formatter(FORMAT)

# Check for IPython and use a special logger
use_ipython_handler = False
try:
    import IPython
except ImportError:
    pass
else:
    if IPython.get_ipython() is not None:
        use_ipython_handler = True
if use_ipython_handler:
    default_handler = IPythonStreamHandler()
else:
    default_handler = logging.StreamHandler(sys.stdout)
default_handler.setLevel(logging.INFO)
default_handler.setFormatter(formatter)


def handle_logging():
    "Send INFO-level log messages to stdout. Do not propagate."
    if use_ipython_handler:
        # Avoid double-printing messages to IPython stderr.
        trackpy.logger.propagate = False
    trackpy.logger.addHandler(default_handler)
    trackpy.logger.setLevel(logging.INFO)


def ignore_logging():
    "Reset to factory default logging configuration; remove trackpy's handler."
    trackpy.logger.removeHandler(default_handler)
    trackpy.logger.setLevel(logging.NOTSET)
    trackpy.logger.propagate = True


def quiet(suppress=True):
    """Suppress trackpy information log messages.

    Parameters
    ----------
    suppress : boolean
        If True, set the logging level to WARN, hiding INFO-level messages.
        If False, set level to INFO, showing informational messages.
    """
    if suppress:
        trackpy.logger.setLevel(logging.WARN)
    else:
        trackpy.logger.setLevel(logging.INFO)

def _pandas_sort_pre_017(df, by, *args, **kwargs):
    """Use sort() to sort a DataFrame"""
    return df.sort(*args, columns=by, **kwargs)

def _pandas_sort_post_017(df, by, *args, **kwargs):
    """
    Use sort_values() to sort a DataFrame
    This raises a ValueError if the given value is both
    a column and an index label, i.e.:
    ValueError: 'frame' is both an index level and a column
    label, which is ambiguous.
    Because we usually sort by columns, we can rename
    the index to supress the ValueError.
    """
    if df.index.name is not None and df.index.name in by:
        df.index.name += '_index'
    return df.sort_values(*args, by=by, **kwargs)

if is_pandas_since_017:
    pandas_sort = _pandas_sort_post_017
else:
    pandas_sort = _pandas_sort_pre_017

def _pandas_iloc_pre_016(df, inds):
    """Workaround for bug, iloc with empty list, in pandas < 0.16"""
    if len(inds) > 0:
        return df.iloc[inds]
    else:
        return df.iloc[:0]

def _pandas_iloc_since_016(df, inds):
    return df.iloc[inds]

if is_pandas_since_016:
    pandas_iloc = _pandas_iloc_since_016
else:
    pandas_iloc = _pandas_iloc_pre_016

def _pandas_rolling_pre_018(df, window, *args, **kwargs):
    """Use rolling_mean() to compute a rolling average"""
    return pd.rolling_mean(df, window, *args, **kwargs)

def _pandas_rolling_since_018(df, window, *args, **kwargs):
    """Use rolling() to compute a rolling average"""
    return df.rolling(window, *args, **kwargs).mean()

if is_pandas_since_018:
    pandas_rolling = _pandas_rolling_since_018
else:
    pandas_rolling = _pandas_rolling_pre_018


def _pandas_concat_post_023(*args, **kwargs):
    """Pass sort = False. Breaks API by not sorting, but we don't care. """
    kwargs.setdefault('sort', False)
    return pd.concat(*args, **kwargs)

if is_pandas_since_023:
    pandas_concat = _pandas_concat_post_023
else:
    pandas_concat = pd.concat


def guess_pos_columns(f):
    """ Guess the position columns from a given feature DataFrame """
    if 'z' in f:
        pos_columns = ['z', 'y', 'x']
    else:
        pos_columns = ['y', 'x']
    return pos_columns


def default_pos_columns(ndim):
    """ Sets the default position column names """
    if ndim < 4:
        return ['z', 'y', 'x'][-ndim:]
    else:
        return map(lambda i: 'x' + str(i), range(ndim))


def default_size_columns(ndim, isotropic):
    """ Sets the default size column names """
    if isotropic:
        return ['size']
    else:
        return ['size_' + cc for cc in default_pos_columns(ndim)]


def is_isotropic(value):
    """ Determine whether all elements of a value are equal """
    if hasattr(value, '__iter__'):
        return np.all(value[1:] == value[:-1])
    else:
        return True


class ReaderCached(object):
    """ Simple wrapper that provides cacheing of image readers """
    def __init__(self, reader):
        self.reader = reader
        self._cache = None
        self._cache_i = None

    def __getitem__(self, i):
        if self._cache_i == i:
            return self._cache.copy()
        else:
            value = self.reader[i]
            self._cache = value.copy()
            return value

    def __repr__(self):
        return repr(self.reader) + "\nWrapped in ReaderCached"

    def __getattr__(self, attr):
        return getattr(self.reader, attr)


def catch_keyboard_interrupt(gen, logger=None):
    """ A generator that stops on a KeyboardInterrupt """
    running = True
    gen = iter(gen)
    while running:
        try:
            yield next(gen)
        except KeyboardInterrupt:
            if logger is not None:
                logger.warn('KeyboardInterrupt')
            running = False
        except StopIteration:
            running = False
        else:
            pass

EXPONENT_EPS_FLOAT64 = np.log(np.finfo(np.float64).eps)
def safe_exp(arr):
    # Calculate exponent, dealing with NaN and Underflow warnings
    result = np.zeros_like(arr)
    result[np.isnan(arr)] = np.nan  # propagate NaNs
    with np.errstate(invalid='ignore'):  # ignore comparison with NaN
        mask = arr > EXPONENT_EPS_FLOAT64
    result[mask] = np.exp(arr[mask])
    return result
