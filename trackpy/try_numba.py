from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

from warnings import warn
import logging

# re-import some builtins for legacy numba versions if future is installed
from six.moves import range
try:
    from __builtin__ import int, round
except ImportError:
    from builtins import int, round

NUMBA_AVAILABLE = False

try:
    import numba
except ImportError:
    message = ("To use numba-accelerated variants of core "
               "functions, you must install numba.")
else:
    v = numba.__version__
    NUMBA_AVAILABLE = True


def nojit(func=None, **kw):
    """Function decorator that does nothing, in the case numba is not available."""
    def return_decorator(func):
        return func
    if func is None:
        return return_decorator
    else:
        return return_decorator(func)


if NUMBA_AVAILABLE:
    try_numba_jit = numba.jit
else:
    try_numba_jit = nojit


def disable_numba():
    """Deprecated. Does nothing."""
    pass


def enable_numba():
    """Deprecated. Does nothing."""
    pass