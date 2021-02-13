import itertools
import warnings
import logging

import numpy as np

from ..utils import guess_pos_columns, validate_tuple, pandas_sort
from .linking import link_iter

logger = logging.getLogger(__name__)


def coords_from_df_partial(df, pos_columns, t_column, link_frame_nos):
    for frame_no in link_frame_nos:
        yield frame_no, df.loc[df[t_column] == frame_no, pos_columns].values


def link_partial(f, search_range, link_range,
                 pos_columns=None, t_column='frame', **kwargs):
    """Patch the trajectories in a DataFrame by linking only a range of frames

    A dataset can be divided into several patches (time intervals) and
    linked separately with this function, for e.g. parallel processing. The
    results can then be stitched back together with reconnect_traj_patch().

    Another application is to link a portion of a dataset again with adjusted
    parameters.

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
    link_range : tuple of 2 ints
        Only frames in the range(start, stop) will be analyzed.
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
    The t_column (by default: 'frame') will be coerced to integer.

    Notes
    -----
    The memory option cannot retain particles at the boundaries between
    patches. Likewise, if a predictor depends on past trajectories, it may
    work poorly at the start of each patch.

    Examples
    --------
    We purposely define a DataFrame with erroneously linked particles:

    >>> df = pd.DataFrame({
    ...     'x': [5., 10., 15., 20., 10., 14., 19.],
    ...     'y': [0., 0., 0, 0., 2., 4., 6.],
    ...     'particle': [5, 5, 8, 8, 8, 5, 5],
    ...     'frame': [0, 1, 2, 3, 1, 2, 3]
    ... })

    We can fix the missed link at frame 1 by calling link_partial like so:

    >>> link_partial(df, search_range=20., link_range=(1, 2))
          x    y  particle  frame
    0   5.0  0.0         5      0
    1  10.0  0.0         5      1
    4  10.0  2.0         8      1
    2  15.0  0.0         5      2
    5  14.0  4.0         8      2
    3  20.0  0.0         5      3
    6  19.0  6.0         8      3

    Note that this partial link remapped the particle indices after the
    link_range. This is done by the function ``reconnect_traj_patch``

    See also
    --------
    reconnect_traj_patch
    """
    if pos_columns is None:
        pos_columns = guess_pos_columns(f)
    ndim = len(pos_columns)
    search_range = validate_tuple(search_range, ndim)
    if kwargs.get('memory', 0) > 0:
        warnings.warn("Particles are not memorized over patch edges.")

    full_range = (int(f[t_column].min()), int(f[t_column].max()) + 1)
    start, stop = link_range
    assert start < stop
    if start is None:
        start = full_range[0]
    elif start < full_range[0]:
        start = full_range[0]

    if stop is None:
        stop = full_range[1]
    elif stop > full_range[1]:
        stop = full_range[1]

    link_frame_nos = range(start, stop)

    # copy the dataframe
    f = f.copy()
    # coerce t_column to integer type
    if not np.issubdtype(f[t_column].dtype, np.integer):
        f[t_column] = f[t_column].astype(np.integer)
    # sort on the t_column
    pandas_sort(f, t_column, inplace=True)

    if 'particle' in f and (start > full_range[0] or stop < full_range[1]):
        f['_old_particle'] = f['particle'].copy()
    elif not 'particle' in f:
        f['particle'] = -1

    coords_iter = coords_from_df_partial(f, pos_columns, t_column,
                                         link_frame_nos)
    for i, _ids in link_iter(coords_iter, search_range, **kwargs):
        f.loc[f[t_column] == i, 'particle'] = _ids

    if '_old_particle' in f:
        reconnect_traj_patch(f, (start, stop), '_old_particle', t_column)
        f.drop('_old_particle', axis=1, inplace=True)

    return f


def reconnect_traj_patch(f, link_range, old_particle_column, t_column='frame'):
    """Reconnect the trajectory inside a range of frames to the trajectories
    outside the range.

    Requires a column with the original particle indices. Does not work in
    combination with memory. Changes the provided DataFrame inplace.

    See also
    --------
    link_partial
    """
    start, stop = link_range
    assert start < stop
    mapping_patch = dict()
    mapping_after = dict()

    # reconnect at first frame_no
    for p_new, p_old in f.loc[f[t_column] == start,
                              ['particle', old_particle_column]].values:
        if p_old < 0:
            continue
        # renumber the track inside the patch to the number before the patch
        mapping_patch[p_new] = p_old

    # reconnect at last frame_no
    for p_new, p_old in f.loc[f[t_column] == stop - 1,
                              ['particle', old_particle_column]].values:
        if p_old < 0:
            continue
        if p_new in mapping_patch:
            # already connected to a track before patch: renumber after the
            # patch
            mapping_after[p_old] = mapping_patch[p_new]
        else:
            # the track is apparently created inside the patch: use the number
            # after the patch
            mapping_patch[p_new] = p_old

    # renumber possible doubles inside the patch
    # the following ids cannot be used as new ids inside the patch:
    in_patch = (f[t_column] >= start) & (f[t_column] < stop)
    remaining = set(f.loc[in_patch, 'particle'].values) - set(mapping_patch)
    if len(remaining) > 0:
        used = set(f.loc[~in_patch, 'particle'].values)
        gen_ids = itertools.filterfalse(lambda x: x in used, itertools.count())

        for p_new, p_mapped in zip(remaining, gen_ids):
            mapping_patch[p_new] = p_mapped

    f.loc[in_patch, 'particle'] = f.loc[in_patch, 'particle'].replace(mapping_patch)
    f.loc[f[t_column] >= stop, 'particle'] = \
        f.loc[f[t_column] >= stop, 'particle'].replace(mapping_after)
