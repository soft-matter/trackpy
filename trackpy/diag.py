from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

from . import __version__
from . import try_numba
from . import preprocessing

def performance_report():
    """Display summary of which optional speedups are installed/enabled"""
    print("Yes, but could it be faster?")
    if try_numba.NUMBA_AVAILABLE:
        #FIXME Handle 0.12/0.11 distinction
        print("FAST: numba is available and enabled "
              "(fast subnets and feature-finding).")
    else:
        print("SLOW: numba was not found")

    if preprocessing.USING_FFTW:
        print("FAST: Using pyfftw for image preprocessing.")
    else:
        print("SLOW: pyfftw not found (slower image preprocessing).")
