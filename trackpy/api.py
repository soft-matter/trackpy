from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import warnings

# pims import is deprecated. We include it here for backwards
# compatibility, but there will be warnings.
import pims as _pims


def _deprecate_pims(call):
    """Wrap a pims callable with a warning that it is deprecated."""
    def deprecated_pims_import(*args, **kw):
        """Class imported from pims package. Its presence in trackpy is deprecated."""
        warnings.warn(('trackpy.{0} is being called, but "{0}" is really part of the '
                      'pims package. It will not be in future versions of trackpy. '
                      'Consider importing pims and calling pims.{0} instead.'
                      ).format(call.__name__), UserWarning)
        return call(*args, **kw)
    return deprecated_pims_import


ImageSequence = _deprecate_pims(_pims.ImageSequence)
Video = _deprecate_pims(_pims.Video)
TiffStack = _deprecate_pims(_pims.TiffStack)


from .motion import *
from .plots import *
from .linking import *
from .filtering import *
from .feature import *
from .preprocessing import bandpass
from .framewise_data import *
from . import utils
from .try_numba import try_numba_autojit, enable_numba, disable_numba
