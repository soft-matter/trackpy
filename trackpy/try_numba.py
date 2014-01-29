import ast
import sys
import inspect
from functools import wraps, partial
import logging


def _hush_llvm():
    # Necessary for current stable release 0.11.
    # Not necessary in master, probably future release will fix.
    # See http://stackoverflow.com/a/20663852/1221924
    import numba.codegen.debug
    llvmlogger = logging.getLogger('numba.codegen.debug')
    llvmlogger.setLevel(logging.INFO)


ENABLE_NUMBA_ON_IMPORT = True
_registered_functions = list()  # functions that can be numba-compiled

try:
    import numba
except ImportError:
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True
    _hush_llvm()


class RegisteredFunction(object):
    "Enable toggling between original function and numba-compiled one."

    def __init__(self, func, fallback=None):
       self.func = func
       self.func_name = func.func_name
       module_name = inspect.getmoduleinfo(func.func_globals['__file__']).name
       module_name = '.'.join(['trackpy', module_name])
       self.module_name = module_name
       if NUMBA_AVAILABLE:
           self.compiled = numba.autojit(func)
       if fallback is not None:
           self.ordinary = fallback
       else:
           self.ordinary = func

    def point_to_compiled_func(self):
        setattr(sys.modules[self.module_name], self.func_name, self.compiled)

    def point_to_ordinary_func(self):
        setattr(sys.modules[self.module_name], self.func_name, self.ordinary)


def try_numba_autojit(func):
    # First, handle optional arguments. This is confusing; see
    # http://stackoverflow.com/questions/3888158
    # TODO

    # Register the function with a global list of numba-enabled functions.
    f = RegisteredFunction(func)
    _registered_functions.append(f)

    if ENABLE_NUMBA_ON_IMPORT and NUMBA_AVAILABLE:
        # Overwrite the function's reference with a numba-compiled function.
        # This can be undone by calling disable_numba()
        return f.compiled
    else:
        return f.ordinary


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
        raise ImportError("To use numba-accelerated variants of core "
                          "functions, you must install numba.")
