from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

import sys
import inspect
from warnings import warn
import logging


def _hush_llvm():
    # Necessary for current stable release 0.11.
    # Not necessary (and unimplemented) in numba >= 0.12 (February 2014)
    # See http://stackoverflow.com/a/20663852/1221924
    try:
        import numba.codegen.debug
        llvmlogger = logging.getLogger('numba.codegen.debug')
        llvmlogger.setLevel(logging.INFO)
    except ImportError:
        pass


ENABLE_NUMBA_ON_IMPORT = True
_registered_functions = list()  # functions that can be numba-compiled

NUMBA_AVAILABLE = False

try:
    import numba
except ImportError:
    message = ("To use numba-accelerated variants of core "
               "functions, you must install numba.")
else:
    v = numba.__version__
    major, minor, micro = v.split('.')
    if major == '0' and minor == '12' and micro == '0':
        # Importing numba code will take forever. Disable numba.
        message = ("Trackpy does not support numba 0.12.0. "
                   "Version {0} is currently installed. Trackpy will run "
                   "with numba disabled. Please downgrade numba to version "
                   "0.11, or update to latest version.".format(v))
        warn(message)
    else:
        NUMBA_AVAILABLE = True
        _hush_llvm()


class RegisteredFunction(object):
    "Enable toggling between original function and numba-compiled one."

    def __init__(self, func, fallback=None, autojit_kw=None):
        self.func = func
        # This covers a Python 2/3 change not covered by six
        try:
            self.func_name = func.__name__
        except AttributeError:
            self.func_name = func.func_name
        module_name = inspect.getmoduleinfo(
            six.get_function_globals(func)['__file__']).name
        module_name = '.'.join(['trackpy', module_name])
        self.module_name = module_name
        self.autojit_kw = autojit_kw
        if fallback is not None:
            self.ordinary = fallback
        else:
            self.ordinary = func

    @property
    def compiled(self):
        # Compile it if this is the first time.
        if (not hasattr(self, '_compiled')) and NUMBA_AVAILABLE:
            if self.autojit_kw is not None:
                self._compiled = numba.autojit(**self.autojit_kw)(self.func)
            else:
                self._compiled = numba.autojit(self.func)
        return self._compiled

    def point_to_compiled_func(self):
        setattr(sys.modules[self.module_name], self.func_name, self.compiled)

    def point_to_ordinary_func(self):
        setattr(sys.modules[self.module_name], self.func_name, self.ordinary)


def try_numba_autojit(func=None, **kw):
    """Wrapper for numba.autojit() that treats the function as pure Python if numba is missing.

    Usage is as with autojit(): Either as a bare decorator (no parentheses), or with keyword
    arguments.

    The resulting compiled numba function can subsequently be turned on or off with
    enable_numba() and disable_numba(). It will be on by default."""
    def return_decorator(func):
        # Register the function with a global list of numba-enabled functions.
        f = RegisteredFunction(func, autojit_kw=kw)
        _registered_functions.append(f)

        if ENABLE_NUMBA_ON_IMPORT and NUMBA_AVAILABLE:
            # Overwrite the function's reference with a numba-compiled function.
            # This can be undone by calling disable_numba()
            return f.compiled
        else:
            return f.ordinary
    if func is None:
        return return_decorator
    else:
        return return_decorator(func)

def disable_numba():
    "Do not use numba-accelerated functions, even if numba is available."
    for f in _registered_functions:
        f.point_to_ordinary_func()


def enable_numba():
    "Use numba-accelerated variants of core functions."
    if NUMBA_AVAILABLE:
        for f in _registered_functions:
            f.point_to_compiled_func()
    else:
        raise ImportError(message)
