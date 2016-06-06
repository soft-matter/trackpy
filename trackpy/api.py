from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import warnings

from .motion import msd, imsd, emsd, compute_drift, subtract_drift, \
           proximity, vanhove, relate_frames, velocity_corr, \
           direction_corr, is_typical, diagonal_size
from .static import proximity, pair_correlation_2d, pair_correlation_3d
from .plots import annotate, annotate3d, plot_traj, ptraj, \
           plot_displacements, subpx_bias, mass_size, mass_ecc, \
           scatter, scatter3d, plot_traj3d, ptraj3d, plot_density_profile
from .linking import HashTable, TreeFinder, Point, PointND, \
           Track, TrackUnstored, UnknownLinkingError, \
           SubnetOversizeException, link, link_df, link_iter, \
           link_df_iter, strip_diagnostics
from .filtering import filter_stubs, filter_clusters, filter
from .feature import locate, batch, percentile_threshold, local_maxima, \
           refine, estimate_mass, estimate_size, minmass_version_change
from .preprocessing import bandpass
from .framewise_data import FramewiseData, PandasHDFStore, PandasHDFStoreBig, \
           PandasHDFStoreSingleNode
from . import utils
from . import artificial
from .utils import handle_logging, ignore_logging, quiet
from .try_numba import try_numba_autojit, enable_numba, disable_numba


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

