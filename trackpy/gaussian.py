from __future__ import (division, print_function, unicode_literals,
                        absolute_import)

import six
import logging
from types import ModuleType

import numpy as np
import pandas as pd

from .preprocessing import lowpass
from .masks import slice_image, binary_mask_multiple
from .utils import guess_pos_columns, validate_tuple

from scipy.optimize import leastsq
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

T_COLUMN = 'frame'
SIGNAL_COLUMN = 'signal'

# make a dynamic module; run imports inside this module
residuals = ModuleType('residuals')
exec("from __future__ import division\nfrom numpy import exp",
     residuals.__dict__)

func_library = dict(
    gauss2D=dict(params=['signal', 'size', 'y', 'x'],
                 func='signal{i} * exp( -((x-x{i})**2 + '
                                         '(y-y{i})**2) / (size{i}**2) )',
                 dfunc=dict(signal='func{i} / signal{i}',
                            size='2*((x-x{i})**2 + (y-y{i})**2) '
                                 '/ size{i}**3*func{i}',
                            y='2*(y-y{i}) / size{i}**2 * func{i}',
                            x='2*(x-x{i}) / size{i}**2 * func{i}')),

    gauss2D_a=dict(params=['signal', 'size_y', 'size_x', 'y', 'x'],
                   func='signal{i} * exp( -(((x-x{i})/size_x{i})**2)'
                        '                 -(((y-y{i})/size_y{i})**2) )',
                   dfunc=dict(signal='func{i} / signal{i}',
                              size_y='2*(y-y{i})**2 / size_y{i}**3 * func{i}',
                              size_x='2*(x-x{i})**2 / size_x{i}**3 * func{i}',
                              y='2*(y-y{i}) / size_y{i}**2 * func{i}',
                              x='2*(x-x{i}) / size_x{i}**2 * func{i}')),


    gauss3D=dict(params=['signal', 'size', 'z', 'y', 'x'],
                 func='signal{i} * exp( -(3/2*((x-x{i})**2 + '
                                              '(y-y{i})**2 + '
                                              '(z-z{i})**2)) / size{i}**2 )',
                 dfunc=dict(signal='func{i} / signal{i}',
                            size='3*((x-x{i})**2 + (y-y{i})**2 + (z-z{i})**2) '
                                 '/ size{i}**3 * func{i}',
                            z='3*(z-z{i}) / size{i}**2 * func{i}',
                            y='3*(y-y{i}) / size{i}**2 * func{i}',
                            x='3*(x-x{i}) / size{i}**2 * func{i}')),

    gauss3D_a=dict(params=['signal', 'size_z', 'size_y', 'size_x',
                           'z', 'y', 'x'],
                   func='signal{i} * exp( -(3/2*((x-x{i})/size_x{i})**2)'
                        '                 -(3/2*((y-y{i})/size_y{i})**2)'
                        '                 -(3/2*((z-z{i})/size_z{i})**2) )',
                   dfunc=dict(signal='func{i} / signal{i}',
                              size_z='3*(z-z{i})**2 / size_z{i}**3 * func{i}',
                              size_y='3*(y-y{i})**2 / size_y{i}**3 * func{i}',
                              size_x='3*(x-x{i})**2 / size_x{i}**3 * func{i}',
                              z='3*(z-z{i}) / size_z{i}**2 * func{i}',
                              y='3*(y-y{i}) / size_y{i}**2 * func{i}',
                              x='3*(x-x{i}) / size_x{i}**2 * func{i}')))


def generate_fit_function(n_func, name, ndim, isotropic, consts=None):
    """Generates code that defines a fitting function that is the addition of
    N functions as defined inside func_dict."""
    name = '{name}{ndim}D'.format(name=name, ndim=ndim)
    if not isotropic:
        name += '_a'
    func_dict = func_library[name]

    if consts is None:
        consts = []
    func_name = name + '_' + '_'.join(consts)

    var_params = []
    const_params = []
    for i in range(n_func):
        for param in func_dict['params']:
            if param in consts:
                const_params.append(param + str(i))
            else:
                var_params.append(param + str(i))

    coords = ['z', 'y', 'x'][-ndim:]
    args = ', '.join(['vars', 'im'] + coords + ['consts'])
    var_params = ', '.join(var_params)
    const_params = ', '.join(const_params)
    func = ' + '.join([func_dict['func'].format(i=i) for i in range(n_func)])

    res = "def {func_name}_{n_func}({args}):"
    res += "\n    {var_params} = vars"
    if len(const_params) > 0:
        res += "\n    {const_params} = consts"
    res += "\n    return {func} - im\n"

    if 'dfunc' in func_dict:
        dfunc_list = []
        for i in range(n_func):
            for param in func_dict['params']:
                if param not in consts:
                    dfunc_list.append(func_dict['dfunc'][param].format(i=i))
        dfunc = ', '.join(dfunc_list)

        res += "\ndef {func_name}_{n_func}_dfunc({args}):"
        res += "\n    {var_params} = vars"
        if len(const_params) > 0:
            res += "\n    {const_params} = consts"
        if 'func0' in dfunc:
            for i in range(n_func):
                res += ("\n    func{i} = " + func_dict['func']).format(i=i)
        res += "\n    return [{dfunc}]"

    return res.format(**locals())


def get_fit_function(n_func, name, ndim, isotropic, consts=None):
    """Checks if the fitting function (=residual) was already generated before,
    if not, generates it and adds it to the residuals library."""
    lib_name = '{name}{ndim}D'.format(name=name, ndim=ndim)
    if not isotropic:
        lib_name += '_a'
    if consts is None:
        consts = []
    residual_name = lib_name + '_' + '_'.join(consts) + '_' + str(n_func)

    if not hasattr(residuals, residual_name):
        exec(generate_fit_function(n_func, name, ndim, isotropic, consts),
             residuals.__dict__)

    dfunc_name = residual_name + '_dfunc'
    if hasattr(residuals, dfunc_name):
        return getattr(residuals, residual_name), getattr(residuals, dfunc_name)
    else:
        return getattr(residuals, residual_name), None


def prepare_subimage(coords, image, radius, noise_size=None, threshold=None):
    # slice region around cluster
    im, origin = slice_image(coords, image, radius)

    # do lowpass filter
    if noise_size is not None:
        if threshold is None:
            threshold = 0
        im = lowpass(im, noise_size, threshold)

    # create the mask
    coords_rel = coords - np.array(origin)[np.newaxis, :]
    mask = binary_mask_multiple(coords_rel, im.shape, radius)

    # create the coordinates
    mesh = np.meshgrid(*[np.arange(o, o+s)
                         for (o, s) in zip(origin, im.shape)], indexing='ij')
    mesh = (m[mask].astype(np.float64) for m in mesh)

    return im[mask].astype(np.float64), tuple(mesh), origin


class Clusters(object):
    def __init__(self, indices):
        self._cl = {i: {i} for i in indices}
        self._f = list(indices)

    @property
    def clusters(self):
        return self._cl

    def __iter__(self):
        return (list(self._cl[k]) for k in self._cl)

    def add(self, a, b):
        i1 = self._f[a]
        i2 = self._f[b]
        if i1 != i2:  # if a and b are already clustered, do nothing
            self._cl[i1] = self._cl[i1].union(self._cl[i2])
            for f in self._cl[i2]:
                self._f[f] = i1
            del self._cl[i2]

    @property
    def cluster_id(self):
        return self._f

    @property
    def cluster_size(self):
        result = [None] * len(self._f)
        for cluster in self:
            for f in cluster:
                result[f] = len(cluster)
        return result


def _find(f, separation):
    """ Find clusters in a list or ndarray of coordinates.

    Parameters
    ----------
    f: iterable of coordinates
    separation: tuple of numbers
        Separation distance below which particles are considered inside cluster

    Returns
    ----------
    ids : ndarray of cluster IDs per particle
    sizes : ndarray of cluster sizes
    """
    # kdt setup
    pairs = cKDTree(np.array(f) / separation).query_pairs(1)

    clusters = Clusters(range(len(f)))
    for (a, b) in pairs:
        clusters.add(a, b)

    return clusters.cluster_id, clusters.cluster_size


def find_iter(f, separation, pos_columns=None, t_column='frame'):
    """ Find clusters in a DataFrame, returns generator iterating over frames.

    Parameters
    ----------
    f: DataFrame
        pandas DataFrame containing pos_columns and t_column
    separation: number or tuple
        Separation distance below which particles are considered inside cluster
    pos_columns: list of strings, optional
        Column names that contain the position coordinates.
        Defaults to ['y', 'x'] (or ['z', 'y', 'x'] if 'z' exists)
    t_column: string
        Column name containing the frame number (Default: 'frame')

    Returns
    ----------
    generator of:
        frame_no
        DataFrame with added cluster and cluster_size column
    """
    if pos_columns is None:
        pos_columns = guess_pos_columns(f)

    next_id = 0

    for frame_no, f_frame in f.groupby(t_column):
        ids, sizes = _find(f_frame[pos_columns].values, separation)
        result = f_frame.copy()
        result['cluster'] = ids
        result['cluster_size'] = sizes
        result['cluster'] += next_id
        next_id = result['cluster'].max() + 1
        yield frame_no, result


def find_clusters(f, separation, pos_columns=None, t_column='frame'):
    """ Find clusters in a DataFrame of points from several frames.

    Parameters
    ----------
    f: DataFrame
        pandas DataFrame containing pos_columns and t_column
    separation: number or tuple
        Separation distance below which particles are considered inside cluster
    pos_columns: list of strings, optional
        Column names that contain the position coordinates.
        Defaults to ['y', 'x'] (or ['z', 'y', 'x'] if 'z' exists)
    t_column: string
        Column name containing the frame number (Default: 'frame')

    Returns
    ----------
    DataFrame
    """
    if t_column not in f:
        f[t_column] = 0
        remove_t_column = True
    else:
        remove_t_column = False

    result = pd.concat((x[1] for x in find_iter(f, separation,
                                                pos_columns, t_column)))

    if remove_t_column:
        del f[t_column]

    return result


def mass_to_max(mass, size, ndim):
    if hasattr(size, '__iter__'):
        assert len(size) == ndim
        return mass / (np.pi * np.prod(size))
    else:
        return mass / (np.pi * size**ndim)


def max_to_mass(max_value, size, ndim):
    if hasattr(size, '__iter__'):
        assert len(size) == ndim
        return max_value * (np.pi * np.prod(size))
    else:
        return max_value * (np.pi * size**ndim)


class RefineException(Exception):
    pass


def _fit_gauss_iter(f, image, diameter, var_size=False, var_signal=False,
                    pos_columns=None, ignore_single=False,
                    noise_size=1, threshold=None, max_iter=10, max_shift=1):
    """ Refines cluster coordinates using gaussian fits, returns generator
    iterating over cluster IDs.

    Parameters
    ----------
    f: DataFrame
        pandas DataFrame containing coordinates of features
        required columns: positions, 'signal', 'size', 'cluster', 'cluster_size'
    image: ndarray
        Image, containing only one channel
    diameter: number or tuple
        Determines mask size that is used on the image before fitting
    var_size: boolean
        Determines whether size is varied in fitting. Default False.
    var_signal: boolean
        Determines whether signal is varied in fitting. Default False.
    pos_columns: list of strings, optional
        Column names that contain the position coordinates.
        Defaults to ['y', 'x'] (or ['z', 'y', 'x'] if 'z' exists)
    ignore_single: boolean
        Determines whether single particles are skipped. Default False.
    noise_size: number
        Noise size used in lowpass filter
    threshold: number
        Threshold used in lowpass filter
    max_iter: int
        Max number of whole-pixel shifts in refine.
    max_shift: float
        Maximum satisfactory out-of-center distance. When the fitted gaussian
        is more out of center, do extra iteration. Default 1.


    Returns
    -------
    iterable of new coordinates per cluster of particles
    """
    ndim = image.ndim
    diameter = validate_tuple(diameter, ndim)
    radius = tuple([x//2 for x in diameter])
    isotropic = np.all(diameter[1:] == diameter[:-1])

    assert SIGNAL_COLUMN in f
    assert 'cluster' in f

    if pos_columns is None:
        pos_columns = ['z', 'y', 'x'][-ndim:]
    for col in pos_columns:
        assert col in f

    if isotropic:
        size_cols = ['size']
    else:
        size_cols = ['size_z', 'size_y', 'size_x'][-ndim:]
    for col in size_cols:
        assert col in f

    var_param_cols = []
    const_param_cols = []
    if var_signal:
        var_param_cols += [SIGNAL_COLUMN]
    else:
        const_param_cols += [SIGNAL_COLUMN]

    if var_size:
        var_param_cols += size_cols
    else:
        const_param_cols += size_cols

    var_param_cols += pos_columns

    for cluster_id, f_cluster in f.groupby('cluster'):
        result = f_cluster.copy()
        if ignore_single and len(f_cluster) == 1:
            raise RefineException
        try:
            consts = (result[const_param_cols].values.ravel(),)
            p0 = result[var_param_cols].values
            if not (np.isfinite(consts[0]).all() and np.isfinite(p0).all()):
                raise RefineException
            for _ in range(max_iter):
                im_masked, mesh, origin = prepare_subimage(p0[:, -ndim:], image,
                                                           radius, noise_size,
                                                           threshold)
                if origin is None:  # coordinates are out of image bounds
                    raise RefineException
                residual, dfunc = get_fit_function(len(result), 'gauss',
                                                   image.ndim, isotropic,
                                                   const_param_cols)

                fit_args = (im_masked,) + mesh + consts
                try:
                    x, cov_x, _, _, ier = leastsq(residual, list(p0.ravel()),
                                                  fit_args, dfunc,
                                                  col_deriv=True,
                                                  full_output=True)
                except ValueError or TypeError:
                    raise RefineException

                if ier in [1, 2, 3] and x is not None and cov_x is not None:
                    new_coords = np.array(x).reshape(p0.shape)
                    std = np.sqrt(np.diag(cov_x)).reshape(p0.shape)
                else:
                    raise RefineException

                # check if found coords are inside the subimage
                upper_bound = [int(m.max()) + 1 for m in mesh]
                if (np.any(new_coords[:, -ndim:] < origin) or
                        np.any(new_coords[:, -ndim:] >= upper_bound)):
                    raise RefineException

                # check if found coords are MAX_SHIFT px from image center.
                shifts = np.abs(p0[:, -ndim:] - new_coords[:, -ndim:])
                if np.sum(shifts**2) < max_shift**2:
                    break  # stop iteration: accept result

                p0 = new_coords.copy()

            if np.any(std[:, -ndim:] > 1):  # more than one pixel variance
                raise RefineException
            if np.any(np.abs(std[:, :-ndim] / new_coords[:, :-ndim]) > 0.1):
                raise RefineException
            if var_size:  # size might be negative: take absolute value here
                new_coords[:, :-ndim] = np.abs(new_coords[:, :-ndim])
        except RefineException:
            result['gaussian'] = False
        else:
            result[var_param_cols] = new_coords
            result['gaussian'] = True

        yield result


def refine_gaussian(f, image, diameter, separation, var_size=False,
                    var_signal=False, pos_columns=None, t_column='frame',
                    ignore_single=False, noise_size=1, threshold=None,
                    max_iter=10, max_shift=1):
    """ Refines coordinates in a single image using gaussian fits.

    Parameters
    ----------
    f: DataFrame
        pandas DataFrame containing coordinates of features
        required columns: positions, 'signal', 'size', 'cluster', 'cluster_size'
    image: ndarray
        Image, containing only one channel
    diameter: number or tuple
        Determines mask size that is used on the image before fitting
    separation: number or tuple
        Separation distance below which particles are considered inside cluster
    var_size: boolean
        Determines whether size is varied in fitting. Default False.
    var_signal: boolean
        Determines whether signal is varied in fitting. Default False.
    pos_columns: list of strings, optional
        Column names that contain the position coordinates.
        Defaults to ['y', 'x'] (or ['z', 'y', 'x'] if 'z' exists)
    t_column: string
        Column name containing the frame number (Default: 'frame')
    ignore_single: boolean
        Determines whether single particles are skipped. Default False.
    noise_size: number
        Noise size used in lowpass filter
    threshold: number
        Threshold used in lowpass filter
    max_iter: int
        Max number of whole-pixel shifts in refine.
    max_shift: float
        Maximum satisfactory out-of-center distance. When the fitted gaussian
        is more out of center, do extra iteration. Default 1.


    Returns
    -------
    copy of input DataFrame, updated with the corrected coordinates
    """
    f = find_clusters(f, separation, pos_columns, t_column)
    return pd.concat(_fit_gauss_iter(f, image, diameter, var_size, var_signal,
                                     pos_columns, ignore_single, noise_size,
                                     threshold, max_iter, max_shift))


def refine_gaussian_batch(f, frames, diameter, separation, var_size=False,
                          var_signal=False, pos_columns=None, t_column='frame',
                          ignore_single=False, noise_size=1, threshold=None,
                          max_iter=10, max_shift=1):
    """ Refines coordinates in a sequence of images using gaussian fits.

    Parameters
    ----------
    f: DataFrame
        pandas DataFrame containing coordinates of features
        required columns: positions, 'signal', 'size'
    image: ndarray
        Image, containing only one channel
    diameter: number or tuple
        Determines mask size that is used on the image before fitting
    separation: number or tuple
        Separation distance below which particles are considered inside cluster
    var_size: boolean
        Determines whether size is varied in fitting. Default False.
    var_signal: boolean
        Determines whether signal is varied in fitting. Default False.
    pos_columns: list of strings, optional
        Column names that contain the position coordinates.
        Defaults to ['y', 'x'] (or ['z', 'y', 'x'] if 'z' exists)
    t_column: string
        Column name containing the frame number (Default: 'frame')
    ignore_single: boolean
        Determines whether single particles are skipped. Default False.
    noise_size: number
        Noise size used in lowpass filter
    threshold: number
        Threshold used in lowpass filter
    max_iter: int
        Max number of whole-pixel shifts in refine.
    max_shift: float
        Maximum satisfactory out-of-center distance. When the fitted gaussian
        is more out of center, do extra iteration. Default 1.

    Returns
    ----------
    copy of input DataFrame, updated with the corrected coordinates
    """
    result = []
    skipped = 0
    for frame_no, f_frame in find_iter(f, separation, pos_columns, t_column):
        f_frame = pd.concat(_fit_gauss_iter(f_frame, frames[frame_no], diameter,
                                            var_size, var_signal, pos_columns,
                                            ignore_single, noise_size,
                                            threshold, max_iter, max_shift))
        result.append(f_frame)

        failed = len(f_frame) - f_frame['gaussian'].sum()
        if ignore_single:
            skipped = np.sum(f_frame['cluster_size'].values == 1)
            failed -= skipped
        logger.info("Frame %d: refined %d of %d features, %d failed",
                    frame_no, len(f_frame) - skipped, len(f_frame), failed)

    return pd.concat(result)
