from __future__ import division, print_function, absolute_import

import six
import logging
from warnings import warn
import numpy as np
from scipy.optimize import minimize

from ..preprocessing import lowpass
from ..static import find_clusters
from ..masks import slice_image
from ..utils import (guess_pos_columns, validate_tuple, is_isotropic,
                     obtain_size_columns, RefineException, ReaderCached,
                     catch_keyboard_interrupt)
from ._fitfunc import FitFunctions, vect_to_params, vect_from_params
from .center_of_mass import center_of_mass as refine_com

try:
    from numdifftools import Hessian
except ImportError:
    Hessian = None

logger = logging.getLogger(__name__)


def _wrap_fun(func, params_const, modes, ids=None):
    def wrapped(vect, *args, **kwargs):
        params = vect_to_params(vect, params_const, modes, ids)
        return func(params, *args, **kwargs)
    return wrapped


def wrap_constraints(constraints, params_const, modes, groups=None):
    if constraints is None:
        return []

    if groups is not None:
        cl_sizes = np.array([len(params_const)], dtype=np.int)

    result = []
    for cons in constraints:
        cluster_size = cons.get('cluster_size', None)
        if cluster_size is None:
            # provide all parameters to the constraint
            def wrapped(vect, *args, **kwargs):
                params = vect_to_params(vect, params_const, modes, groups)
                return cons['fun'](params[np.newaxis, :, :], *args, **kwargs)
        elif groups is None:
            if len(params_const) != cluster_size:
                continue
            # provide all parameters to the constraint
            def wrapped(vect, *args, **kwargs):
                params = vect_to_params(vect, params_const, modes, groups)
                return cons['fun'](params[np.newaxis, :, :], *args, **kwargs)
        elif cluster_size in cl_sizes:
            groups_this = groups[0][cl_sizes == cluster_size]
            if len(groups_this) == 0:
                continue
            # group the appropriate clusters together and return multiple values
            def wrapped(vect, *args, **kwargs):
                params = vect_to_params(vect, params_const, modes, groups)
                params_grouped = np.array([params[g] for g in groups_this])
                return cons['fun'](params_grouped, *args, **kwargs)
        else:
            continue
        cons_wrapped = cons.copy()
        cons_wrapped['fun'] = wrapped
        result.append(cons_wrapped)
        if 'jac' in cons_wrapped:
            warn('Constraint jacobians are not implemented')
            del cons_wrapped['jac']
    return result


def _dimer_fun(x, dist, ndim):
    pos = x[..., 2:2+ndim]  # get positions only
    return 1 - np.sum(((pos[:, 0] - pos[:, 1])/dist)**2, axis=1)


def dimer(dist, ndim=2):
    """Constrain clusters of 2 at given distance.

    Allows image anisotropy by providing a tuple to as distance"""
    dist = np.array(validate_tuple(dist, ndim))
    return (dict(type='eq', cluster_size=2, fun=_dimer_fun, args=(dist, ndim)),)


def _trimer_fun(x, dist, ndim):
    x = x[..., 2:2+ndim]  # get positions only
    return np.concatenate((1 - np.sum(((x[:, 0] - x[:, 1])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 1] - x[:, 2])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 0] - x[:, 2])/dist)**2, axis=1)))


def trimer(dist, ndim=2):
    """Constrain clusters of 3 at given distance.

    Allows image anisotropy by providing a tuple to as distance.
    Constraints all 3 distances to the same distance."""
    dist = np.array(validate_tuple(dist, ndim))
    return (dict(type='eq', cluster_size=3, fun=_trimer_fun, args=(dist, ndim)),)


def _tetramer_fun_2d(x, dist):
    x = x[..., 2:4]  # get positions only
    dists = np.vstack((np.sum(((x[:, 0] - x[:, 1])/dist)**2, axis=1),
                       np.sum(((x[:, 1] - x[:, 2])/dist)**2, axis=1),
                       np.sum(((x[:, 0] - x[:, 2])/dist)**2, axis=1),
                       np.sum(((x[:, 1] - x[:, 3])/dist)**2, axis=1),
                       np.sum(((x[:, 0] - x[:, 3])/dist)**2, axis=1),
                       np.sum(((x[:, 2] - x[:, 3])/dist)**2, axis=1)))
    # take the 4 smallest: they should be 1
    # do not test the other 2: they are fixed by the 4 first constraints.
    dists = np.sort(dists, axis=0)[:4]
    return np.ravel(1 - dists)


def _tetramer_fun_3d(x, dist):
    x = x[..., 2:5]  # get positions only
    return np.concatenate((1 - np.sum(((x[:, 0] - x[:, 1])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 1] - x[:, 2])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 0] - x[:, 2])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 1] - x[:, 3])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 0] - x[:, 3])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 2] - x[:, 3])/dist)**2, axis=1)))

def tetramer(dist, ndim=2):
    """Constrain clusters of 4 at given distance.

    For 2D: features are in a perfect square (4 constraints)
    For 3D: features are constrained in a tetrahedron (6 constraints).
    Allows image anisotropy by providing a tuple to as distance."""
    dist = np.array(validate_tuple(dist, ndim))
    if ndim == 2:
        return (dict(type='eq', cluster_size=4, fun=_tetramer_fun_2d, args=(dist,)),)
    elif ndim == 3:
        return (dict(type='eq', cluster_size=4, fun=_tetramer_fun_3d, args=(dist,)),)
    else:
        raise NotImplementedError


def _dimer_fun_global(x, mpp, ndim):
    if x.ndim == 2 or len(x) <= 1:
        return []
    pos = x[..., 2:2+ndim]  # get positions only, shape (n_clusters, 2, ndim)
    dist_squared = np.sum(((pos[:, 0] - pos[:, 1])*mpp)**2, axis=1)**2
    return np.diff(dist_squared)


def dimer_global(mpp, ndim=2):
    """Constrain clusters of 2 to a constant, unknown distance.

    Allows image anisotropy by providing ``mpp``, microns per pixel. The
    number of constraints equals the number of frames - 1."""
    # the jacobian seems to slow things down.
    # in tests: 26 iterations without, 198 with
    mpp = np.array(validate_tuple(mpp, ndim))
    return (dict(type='eq', fun=_dimer_fun_global, args=(mpp, ndim,)),)


def prepare_subimage(coords, image, radius, noise_size=None, threshold=None):
    ndim = image.ndim
    radius = validate_tuple(radius, ndim)
    # slice region around cluster
    im, origin = slice_image(coords, image, radius)
    if origin is None:   # coordinates are out of image bounds
        raise RefineException

    # do lowpass filter
    if noise_size is not None:
        if threshold is None:
            threshold = 0
        im = lowpass(im, noise_size, threshold)

    # include the edges where dist == 1 exactly
    dist = [(np.sum(((np.indices(im.shape).T - (coord - origin)) / radius)**2, -1) <= 1)
            for coord in coords]

    # to mask the image
    mask_total = np.any(dist, axis=0).T
    # to mask the masked image
    masks_singles = np.empty((len(coords), mask_total.sum()), dtype=np.bool)
    for i, _dist in enumerate(dist):
        masks_singles[i] = _dist.T[mask_total]

    # create the coordinates
    mesh = np.indices(im.shape, dtype=np.float64)[:, mask_total]
    # translate so that coordinates are in image coordinates
    mesh += np.array(origin)[:, np.newaxis]

    return im[mask_total].astype(np.float64), mesh, masks_singles


def prepare_subimages(coords, groups, frame_nos, reader, radius,
                      noise_size=None, threshold=None):
    # fast shortcut
    if groups is None:
        image, mesh, mask = prepare_subimage(coords, reader[frame_nos[0]],
                                             radius, noise_size, threshold)
        return [image], [mesh], [mask]

    images = []
    meshes = []
    masks = []
    for cl_inds in groups[0]:
        frame_no = frame_nos[cl_inds[0]]
        image, mesh, mask = prepare_subimage(coords[cl_inds], reader[frame_no],
                                             radius, noise_size, threshold)
        images.append(image)
        meshes.append(mesh)
        masks.append(mask)
    return images, meshes, masks


def refine_leastsq(f, reader, diameter, separation=None, fit_function='gauss',
                   param_mode=None, param_val=None, constraints=None,
                   bounds=None, pos_columns=None, t_column='frame',
                   noise_size=None, threshold=None, max_iter=10, max_shift=1,
                   max_rms_dev=1., residual_factor=100000.,
                   compute_error=False, **kwargs):
    """ Refines cluster coordinates by least-squares fitting to radial model
    functions.

    This does not raise an error if minimization failes. Instead, coordinates
    are unchanged and the added column ``cost`` will be ``NaN``.

    Parameters
    ----------
    f : DataFrame
        pandas DataFrame containing coordinates of features.
        Required columns are positions. Any parameter that is not present should
        be either given in the ``param_val`` dict (e.g. 'signal', 'size') or be
        present in the ``default`` field of the used model function.
    reader : pims.FramesSequence or ndarray
        FrameSequence: object that returns an image when indexed. It should
        provide the ``frame_shape`` attribute. If a FrameSequence is not given,
        a single image is assumed and all features that are present in ``f`` are
        assumed to be in that image.
    diameter : number or tuple
        Determines the feature mask size that is used for the refinement
    separation : number or tuple
        Determines the distance below which features are considered in the same
        cluster. By default, equals ``diameter``. As the model feature function is
        only defined up to ``diameter``, it only makes sense to change this
        parameter in some edge cases.
    fit_function : {'gauss', 'ring', 'inv_series_<number>'} or dict, optional
        The shape of the used radial model function. Defaults to Gaussian.
        The ring model function is a displaced gaussian with parameter ``thickness``.
        The ``inv_series_<number>`` model function is the inverse of an
        even polynomial containing ``<number>`` parameters
        (e.g. A / (1 + a r^2 + b r^4 + c r^2 + ...)

        Define your own model function with a dictionary, containing:

            params : list of str
                list of parameter names. has the same length as the ``p`` array
                in ``func`` and ``dfunc``
            func : callable
                The image model function. It takes arguments ``(r2, p, ndim)``,
                with ``r2`` being a 1d ndarray containing the squared reduced
                radial distances.
                (for isotropic, 2D: ``((x - c_x)^2 + (y - c_y)^2) / size^2``).
                ``p`` is a 1d array of parameters single feature parameters.
                ``ndim`` is the number of dimensions. Returns a 1d vector of the
                same size as ``r2`` containing intensity values, normalized to 1.
            dfunc : callable, optional
                takes the same arguments as ``func``. Returns a tuple of size 2.
                The first element: again the image model function, exactly as
                returned by ``func``, because of performance considerations.
                The second element: the Jacobian of ``func``. List of 1d arrays,
                with length ``len(params) + 1``.
                The first element is the derivative w.r.t. `r^2`, the following
                elements each w.r.t. to a custom params, in the order given by
                ``params``.
            default : dict
                Default parameter values.
    param_mode : dict, optional
        For each parameter, define whether it is constant or varies. Also define
        whether variables are equal within each cluster or equal for all features.

        Each parameter can have one of the following values:
            ``'var'`` :
                the parameter is allowed to vary for each feature
                independently
            ``'const'`` :
                the parameter is not allowed to vary
            ``'cluster'`` :
                the parameter is allowed to vary, but is equal
                within each cluster
            ``'global'`` :
                the parameter is allowed to vary, but is equal for each feature
        Not implemented yet: ``'frame'`` and ``'particle'``

        Default values for position coordinates and signal is ``'var'``, for
        background ``'cluster'`` and for all others ``'const'``. Background
        cannot vary per feature.
    param_val : dict, optional
        Default parameter values.
    constraints : iterable of dicts
        Contains definition
        These are described as follows (adapted from scipy.optimize.minimize):

            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint. The function is provided
                a 3d ndarray with on the axes (<cluster>, <feature>, <parameters>)
                parameters are (background, signal, <pos>, <size>, <other>)
            args : sequence, optional
                Extra arguments to be passed to the function.
            cluster_size : integer
                Size of the cluster to which the constraint applies

        Equality constraint means that the constraint function result is to
        be zero whereas inequality means that it is to be non-negative.
    bounds: dict
        Bounds on parameters, in the following forms:
            - Absolute bounds ``{'x': [low, high]}``
            - Difference bounds, one-sided ``{'x_diff': max_diff}``
            - Difference bounds, two-sided ``{'x_diff': [max_diff_below,max_diff_above]}``
            - Relative bounds, one-sided ``{'x_rel_diff': max_fraction_below}``
            - Relative bounds, two-sided ``{'x_rel_diff': [max_fraction_below, max_fraction_above]}``
        When the keyword `pos` is used, this will be distributed to all
        pos_columns (but direct values of each positions will have precedence)
        When the keyword `size` is used, this will be distributed to all sizes,
        in the case of anisotropic sizes (also, direct values have precedence)

        For example, ``{'x': (2, 6), 'x_diff': (5, 5), 'x_rel_diff': 0.2``
        would limit the parameter ``'x'`` between 2 and 6, between ``x-5`` and
        ``x+5``,  and between ``x*(1 - 0.2)`` and ``x*(1 + 0.2)``. The most
        narrow bound is taken.
    pos_columns: list of strings, optional
        Column names that contain the position coordinates.
        Defaults to ['y', 'x'] (or ['z', 'y', 'x'] if 'z' exists)
    t_column: string, optional
        Column name that denotes the frame index. Default 'frame'
    noise_size: number, optional
        Noise size used in lowpass filter. Default None.
    threshold: number, optional
        Threshold used in lowpass filter. Default None.
    max_iter: int, optional
        Max number of whole-pixel shifts in refine. Default 10.
    max_shift: float, optional
        Maximum satisfactory out-of-center distance. When the fitted gaussian
        is more out of center, do extra iteration. Default 1.
    max_rms_dev : float, optional
        Maximum root mean squared difference between the final fit and the
        (preprocessed) image, in units of the image maximum value. Default 1.
    residual_factor : float, optional
        Factor with which the residual is multiplied, something internal inside
        SLSQP makes it work best with this set around 100000. (which is Default)
    compute_error : boolean, optional
        Requires numdifftools to be installed. Default False.
        This is an experimental and untested feature that estimates the error
        in the optimized parameters on a per-feature basis from the curvature
        (diagonal elements of the Hessian) of the objective function in the
        optimized point.
    kwargs : optional
        other arguments are passed directly to scipy.minimize. Defaults are
        ``dict(method='SLSQP', tol=1E-6,
          options=dict(maxiter=100, disp=False))``


    Returns
    -------
    DataFrame of refined coordinates.
    added column 'cluster': the cluster id of the feature.
    added column 'cluster_size': the size of the cluster to which the feature belongs
    added column 'cost': root mean squared difference between the final fit and
        the (preprocessed) image, in units of the cluster maximum value. If the
        optimization fails, no error is raised feature fields are unchanged,
        and this field becomes NaN.\
    addded columns of variable parameters ('x_std', etc.) (only if compute_error is true)
    """
    _kwargs = dict(method='SLSQP', tol=1E-6,
                   options=dict(maxiter=100, disp=False))
    _kwargs.update(kwargs)
    # Initialize variables
    if pos_columns is None:
        pos_columns = guess_pos_columns(f)
    if compute_error and (Hessian is None):
        raise ImportError('compute_error requires the package numdifftools')

    # Cache images
    try:
        # assume that the reader is a FramesSequence
        ndim = len(reader.frame_shape)
        logging = True
    except AttributeError:
        try:
            ndim = reader.ndim
        except AttributeError:
            raise ValueError('For multiple frames, the reader should be a'
                             'FramesSequence object exposing the "frame_shape"'
                             'attribute')
        # reader is a single Frame, wrap it using some logic for the frame_no
        frame_no = None
        if hasattr(reader, 'frame_no'):
            if reader.frame_no is not None:
                frame_no = int(reader.frame_no)

        if frame_no is not None and t_column in f:
            assert np.all(f['frame'] == frame_no)
            reader = {frame_no: reader}
        elif frame_no is not None:
            reader = {frame_no: reader}
            f[t_column] = frame_no
        elif frame_no is None and t_column in f:
            assert f[t_column].nunique() == 1
            reader = {int(f[t_column].iloc[0]): reader}
        else:
            f[t_column] = 0
            reader = {0: reader}
        logging = False

    assert ndim == len(pos_columns)

    diameter = validate_tuple(diameter, ndim)
    radius = tuple([x//2 for x in diameter])
    isotropic = is_isotropic(diameter)
    if separation is None:
        separation = diameter

    ff = FitFunctions(fit_function, ndim, isotropic, param_mode)

    if constraints is None:
        constraints = dict()

    # makes a copy
    f = find_clusters(f, separation, pos_columns, t_column)

    # Assign param_val to dataframe
    if param_val is not None:
        for col in param_val:
            f[col] = param_val[col]
    col_missing = set(ff.params) - set(f.columns)
    for col in col_missing:
        f[col] = ff.default[col]

    if compute_error:
        cols_std = []
        for param in ff.params:
            if ff.param_mode[param] > 0:
                cols_std.append('{}_std'.format(param))
                f[cols_std[-1]] = np.nan
        modes_std = [mode for mode in ff.modes if mode > 0]

    bounds = ff.validate_bounds(bounds, radius=radius)

    # split the problem into smaller ones, depending on param_mode
    modes = np.array(ff.modes)
    if np.any(modes == 2):
        level = 'global'
        # there are globals, we cannot split the problem
        iterable = [(None, f)]
        id_names = ['cluster']
        frames = dict()
        norm = 0.
        for i in f[t_column].unique():
            i = int(i)
            frame = reader[i]
            frames[i] = frame
            norm = max(norm, float(frame.max()))
        norm = norm**2 / residual_factor
        logger.info("Cached all frames")
    elif np.all(modes <= 3):
        level = 'cluster'
        # no globals, no per particle / per frame
        iterable = f.groupby(['frame', 'cluster'])  # ensure sorting per frame
        id_names = None
        frames = ReaderCached(reader)  # cache the last frame
    else:
        raise NotImplemented()

    last_frame = None  # just for logging
    for _, f_iter in catch_keyboard_interrupt(iterable, logger=logger):
        # extract the initial parameters from the dataframe
        params = f_iter[ff.params].values
        if id_names is None:
            groups = None
        else:
            f_iter_temp = f_iter.reset_index()
            groups = [list(f_iter_temp.groupby(col).indices.values()) for col in id_names]
        frame_nos = f_iter[t_column].values

        if level != 'global':
            norm = float(frames[frame_nos[0]].max()) ** 2 / residual_factor
        try:
            if not np.isfinite(params).all():
                raise RefineException
            # extract the coordinates from the parameter array
            coords = params[:, 2:2+ndim]
            # transform the params into a vector for leastq optimization
            vect = vect_from_params(params, ff.modes, groups, operation=np.mean)

            f_constraints = wrap_constraints(constraints, params, ff.modes, groups)
            f_bounds = ff.compute_bounds(bounds, params, groups)
            for _n_iter in range(max_iter):
                sub_images, meshes, masks = prepare_subimages(coords, groups,
                                                              frame_nos, frames,
                                                              radius, noise_size,
                                                              threshold)
                residual, jacobian = ff.get_residual(sub_images, meshes, masks,
                                                     params, groups, norm)

                result = minimize(residual, vect, bounds=f_bounds,
                                  constraints=f_constraints, jac=jacobian,
                                  **_kwargs)
                if not result['success']:
                    raise RefineException(result['message'])

                rms_dev = np.sqrt(result['fun'] / residual_factor)
                params = vect_to_params(result['x'], params, ff.modes, groups)

                # check if found coords are MAX_SHIFT px from image center.
                new_coords = params[:, 2:2+ndim]
                if np.all(np.sum((new_coords - coords)**2, 1) < max_shift**2):
                    break  # stop iteration: accept result

                # set-up for next iteration
                coords = new_coords

            # check the final difference between fit and image
            if rms_dev > max_rms_dev:
                raise RefineException('The rms deviation of the fit ({0:.4f} is'
                                      'more than the maximum value of '
                                      '{1:.4f}.'.format(rms_dev, max_rms_dev))

            # estimate the errors using the Hessian matrix
            # see Bevington PR, Robinson DK (2003) Data reduction and error
            # analysis for the physical sciences (McGraw-Hill Higher Education).
            # 3rd Ed. , equation (8.11)
            if compute_error:
                hessian = Hessian(residual)(result['x'])
                result_std = np.sqrt(2 * np.diag(np.linalg.inv(hessian)))
                params_std = vect_to_params(result_std,
                                            np.empty((len(params),
                                                      len(modes_std))),
                                            modes_std, groups)

        except RefineException as e:
            if level == 'global':
                f['cost'] = np.nan
                if compute_error:
                    f[cols_std] = np.nan
            else:
                f.loc[f_iter.index, 'cost'] = np.nan
                if compute_error:
                    f[f_iter.index, cols_std] = np.nan
            logger.warn('RefineException: ' + e.args[0])
            status = 'failed'
        else:
            if level == 'global':
                f[ff.params] = params
                f['cost'] = rms_dev
                if compute_error:
                    f[cols_std] = params_std
            else:
                f.loc[f_iter.index, ff.params] = params
                f.loc[f_iter.index, 'cost'] = rms_dev
                if compute_error:
                    f.loc[f_iter.index, cols_std] = params_std
            status = 'success'

        if level == 'global':
            logger.info("Global refine {status}: {n} "
                        "features.".format(status=status, n=len(f)))
        elif level == 'cluster':
            cluster_id = int(f_iter['cluster'].iloc[0])
            logger.debug("Refine per cluster {status} in frame {frame_no}, "
                         "cluster {cluster_id} of size "
                         "{cluster_size}".format(status=status,
                                                 frame_no=frame_nos[0],
                                                 cluster_id=cluster_id,
                                                 cluster_size=len(f_iter)))
            if frame_nos[0] != last_frame:
                last_frame = frame_nos[0]
                mesg = "Finished refine per cluster in frame " \
                       "{frame_no}".format(frame_no=last_frame)
                if logging:
                    logger.info(mesg)
                else:
                    logger.debug(mesg)

    return f


def train_leastsq(f, reader, diameter, separation, fit_function,
                  param_mode=None, tol=1e-7, pos_columns=None, **kwargs):
    """Obtain fit parameters from an image of features with equal size.
    Different signal intensities per feature are allowed."""
    try:
        ndim = len(reader.frame_shape)
    except AttributeError:
        ndim = reader.ndim
        reader = [reader]
        if 'frame' in f:
            assert np.all(f['frame'].nunique() == 0)
        else:
            f['frame'] = 0
    diameter = validate_tuple(diameter, ndim)
    radius = tuple([d // 2 for d in diameter])
    if pos_columns is None:
        pos_columns = guess_pos_columns(f)
    isotropic = is_isotropic(diameter)
    size_columns = obtain_size_columns(isotropic, pos_columns)

    # first, refine using center-of-mass
    for frame_no, f_frame in f.groupby('frame'):
        coords = f_frame[pos_columns].values
        image = reader[frame_no]
        tp_result = refine_com(image, image, radius, coords)
        pos = tp_result[:, ndim-1:None:-1]
        if isotropic:
            size = tp_result[:, ndim + 1]
            signal = tp_result[:, ndim + 3]
        else:
            size = tp_result[:, ndim + 1:2*ndim + 1]
            signal = tp_result[:, 2*ndim + 2]
        f.loc[f_frame.index, pos_columns] = pos
        f.loc[f_frame.index, 'signal'] = signal
        f.loc[f_frame.index, size_columns] = size

    if param_mode is None:
        param_mode = dict()

    ff = FitFunctions(fit_function, ndim, isotropic)
    for param in ff.params:
        if param in param_mode:
            continue
        if param in pos_columns + ['signal', 'background']:
            param_mode[param] = 'const'
        else:
            param_mode[param] = 'global'

    bounds = kwargs.pop('bounds', dict())
    if bounds is None:
        bounds = dict()
    for size_col in size_columns:
        if size_col + '_rel_diff' not in bounds:
            bounds[size_col + '_rel_diff'] = (0.9, 9)  # - 90%, +900%

    f = refine_leastsq(f, reader, diameter, separation,
                       fit_function=fit_function, param_mode=param_mode,
                       tol=tol, bounds=bounds, **kwargs)
    assert np.isfinite(f['cost']).all()

    return {p: f[p].mean() for p in param_mode if param_mode[p] == 'global'}
