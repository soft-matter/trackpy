import sys
import warnings

# re-import some builtins for legacy numba versions if future is installed
try:
    from __builtin__ import int, round
except ImportError:
    from builtins import int, round

ENABLE_NUMBA_ON_IMPORT = True
_registered_functions = list()  # functions that can be numba-compiled

NUMBA_AVAILABLE = True
message = ''

try:
    # numba deprecationwarnings from numpy 1.20
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="numpy")
        import numba
except ImportError:
    NUMBA_AVAILABLE = False
    message = ("To use numba-accelerated variants of core "
               "functions, you must install numba.")


class RegisteredFunction:
    """Enable toggling between original function and numba-compiled one."""

    def __init__(self, func, fallback=None, jit_kwargs=None):
        self.func = func
        self.func_name = func.__name__
        self.module_name = func.__module__
        self.jit_kwargs = jit_kwargs
        if fallback is not None:
            self.ordinary = fallback
        else:
            self.ordinary = func

    @property
    def compiled(self):
        # Compile it if this is the first time.
        if (not hasattr(self, '_compiled')) and NUMBA_AVAILABLE:
            if self.jit_kwargs is not None:
                self._compiled = numba.jit(**self.jit_kwargs)(self.func)
            else:
                self._compiled = numba.jit(self.func)
        return self._compiled

    def point_to_compiled_func(self):
        setattr(sys.modules[self.module_name], self.func_name, self.compiled)

    def point_to_ordinary_func(self):
        setattr(sys.modules[self.module_name], self.func_name, self.ordinary)


def try_numba_jit(func=None, **kwargs):
    """Wrapper for numba.jit() that treats the function as pure Python if numba is missing.

    Usage is as with jit(): Either as a bare decorator (no parentheses), or with keyword
    arguments.

    The resulting compiled numba function can subsequently be turned on or off with
    enable_numba() and disable_numba(). It will be on by default."""
    def return_decorator(func):
        # Register the function with a global list of numba-enabled functions.
        f = RegisteredFunction(func, jit_kwargs=kwargs)
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
    """Do not use numba-accelerated functions, even if numba is available."""
    for f in _registered_functions:
        f.point_to_ordinary_func()


def enable_numba():
    """Use numba-accelerated variants of core functions."""
    if NUMBA_AVAILABLE:
        for f in _registered_functions:
            f.point_to_compiled_func()
    else:
        raise ImportError(message)
