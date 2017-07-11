from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from six.moves import range, zip
import warnings
import logging

import numpy as np

from ..utils import guess_pos_columns, validate_tuple, pandas_sort
from .linking import link_iter

logger = logging.getLogger(__name__)


def coords_from_df_partial(df, pos_columns, t_column, link_frame_inds):
    for frame_no in link_frame_inds:
        yield frame_no, df.loc[df[t_column] == frame_no, pos_columns].values


def link_partial(f, search_range, link_range=None,
                 pos_columns=None, t_column='frame', **kwargs):
    """Link a DataFrame of coordinates into trajectories.

    Parameters
    ----------
    f : DataFrame
        The DataFrame must include any number of column(s) for position and a
        column of frame numbers. By default, 'x' and 'y' are expected for
        position, and 'frame' is expected for frame number. See below for
        options to use custom column names.
    search_range : float or tuple
        the maximum distance features can move between frames,
        optionally per dimension
    link_range : range, optional
        Defines the frame numbers that will be analyzed. Default is everything.
    memory : integer, optional
        the maximum number of frames during which a feature can vanish,
        then reappear nearby, and be considered the same particle. 0 by default.
    pos_columns : list of str, optional
        Default is ['y', 'x'], or ['z', 'y', 'x'] when 'z' is present in f
    t_column : str, optional
        Default is 'frame'
    predictor : function, optional
        Improve performance by guessing where a particle will be in
        the next frame.
        For examples of how this works, see the "predict" module.
    adaptive_stop : float, optional
        If not None, when encountering an oversize subnet, retry by progressively
        reducing search_range until the subnet is solvable. If search_range
        becomes <= adaptive_stop, give up and raise a SubnetOversizeException.
    adaptive_step : float, optional
        Reduce search_range by multiplying it by this factor.
    link_strategy : {'recursive', 'nonrecursive', 'numba', 'drop', 'auto'}
        algorithm used to resolve subnetworks of nearby particles
        'auto' uses numba if available
        'drop' causes particles in subnetworks to go unlinked

    Returns
    -------
    DataFrame with added column 'particle' containing trajectory labels.
    The t_column (by default: 'frame') will be coerced to integer."""
    if pos_columns is None:
        pos_columns = guess_pos_columns(f)
    ndim = len(pos_columns)
    search_range = validate_tuple(search_range, ndim)

    reconnect = False
    full_range = (int(f[t_column].min()), int(f[t_column].max()))
    if link_range is None:
        link_range = range(full_range[0], full_range[1] + 1)
    else:
        first, last = link_range.start, link_range.stop - 1
        if first > last:
            first, last = last, first
        if (first > full_range[0] or last < full_range[1]) \
                and 'particle' in f:
            reconnect = True
            f['_old_particle'] = f['particle'].copy()

    # copy the dataframe
    f = f.copy()
    # coerce t_column to integer type
    if not np.issubdtype(f[t_column].dtype, np.integer):
        f[t_column] = f[t_column].astype(np.integer)
    # sort on the t_column
    pandas_sort(f, t_column, inplace=True)

    if not f.index.is_unique:
        f.reset_index(inplace=True, drop=True)
    if not 'particle' in f:
        f['particle'] = -1

    coords_iter = coords_from_df_partial(f, pos_columns, t_column, link_range)
    for i, _ids in link_iter(coords_iter, search_range, **kwargs):
        f.loc[f[t_column] == i, 'particle'] = _ids

    if reconnect:  # reconnect when linking only a patch
        last = i
        mapping = dict()
        # make sure all features are connected properly
        for p_new, p_old in f.loc[f['frame'] == last,
                                  ['particle', '_old_particle']].values:
            if p_old < 0 or p_old == p_new:
                continue
            mapping[p_old] = p_new

        if len(mapping) > 0:
            if first < last:
                range_after = f['frame'] > last
            else:
                range_after = f['frame'] < last
            f.loc[range_after, 'particle'] = \
                  f.loc[range_after, 'particle'].replace(mapping)
        f.drop('_old_particle', axis=1, inplace=True)

    return f
