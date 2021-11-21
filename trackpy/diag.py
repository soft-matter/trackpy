import sys
import importlib
from collections import OrderedDict

from . import try_numba
from . import preprocessing
from . import __version__


def performance_report():
    """Display summary of which optional speedups are installed/enabled"""
    print("Yes, but could it be faster?")
    if try_numba.NUMBA_AVAILABLE:
        print("FAST: numba is available and enabled "
              "(fast subnets and feature-finding).")
    else:
        print("SLOW: numba was not found")


def dependencies():
    """
    Give the version of each of the dependencies -- useful for bug reports.

    Returns
    -------
    result : dict
        mapping the name of each package to its version string or, if an
        optional dependency is not installed, None
    """
    packages = ['numpy', 'scipy', 'matplotlib', 'pandas',
                'sklearn', 'pyyaml', 'tables', 'numba', 'pims']
    result = OrderedDict()

    # trackpy itself comes first
    result['trackpy'] = __version__

    for package_name in packages:
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            result[package_name] = None
        else:
            result[package_name] = package.__version__

    # Build Python version string
    version_info = sys.version_info
    version_string = '.'.join(map(str, [version_info[0], version_info[1],
                                  version_info[2]]))
    result['python'] = version_string

    return result
