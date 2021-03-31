import logging
import warnings
import numpy as np
from scipy.optimize import minimize

from ..static import cluster
from ..masks import slice_image
from ..utils import (guess_pos_columns, validate_tuple, is_isotropic, safe_exp,
                     ReaderCached, default_pos_columns, default_size_columns,
                     is_scipy_15)
from .center_of_mass import refine_com

try:
    from numdifftools import Hessian
except ImportError:
    Hessian = None

logger = logging.getLogger(__name__)

MODE_DICT = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
             'const': 0, 'var': 1, 'global': 2, 'cluster': 3}
# unimplemented modes: {'particle': 4, 'frame': 5}


class RefineException(Exception):
    pass


def vect_from_params(params, modes, groups=None, operation=None):
    """Convert an array of per-feature parameters into a vector
    for least squares optimization

    Parameters
    ----------
    params : 2d ndarray of floats
        The array of per-feature parameters. On the axes are
        (<feature>, <parameter>). Parameters are
        (background, signal, <pos>, <size>, <other>).
    modes: ndarray of integers
        modes of every variable in the array of per-feature parameters:
        - 0 corresponds to constant
        - 1 corresponds to varying
        - 2 corresponds to varying, but equal for each feature
        - 3 corresponds to varying, but equal within cluster
        others are custom (e.g. per particle, per frame)
    groups : iterable of lists of lists
        Nested lists of integers defining groups of parameters. ``groups[0]``
        corresponds to groups of indices defining the clusters. Other elements
        correspond to custom parameter modes (index = n_mode - 3). If the only
        existing modes are 0, 1, and 3, this parameter can be omitted, as
        minimization will be grouped into clusters anyway.
    operation : function
        function that converts a 1d array of parameters into a scalar
        Default None: take the first one (assume they are all equal)

    Returns
    -------
    Vector (1d ndarray) containing the parameters for leastsq optimization.

    See also
    --------
    refine_leastsq
    """
    n, n_vars = params.shape
    assert len(modes) == n_vars
    assert min(modes) >= 0

    result = []
    for i, mode in enumerate(modes):
        mode = int(mode)
        if mode == 0:
            continue  # skip: it is constant
        elif mode == 1:
            # take all
            result.append(params[:, i])
        elif mode == 2 or groups is None:
            # take only one
            if operation is None:
                result.append([params[0, i]])
            else:
                result.append([operation(params[:, i])])
        else:
            try:
                groups_this = groups[mode - 3]
            except (IndexError, TypeError):
                raise ValueError("The groups for mode {} were not provided".format(mode))
            if operation is None:
                # take the first for every unique id
                indices = [g[0] for g in groups_this]
                result.append(params[indices, i])
            else:
                group_vect = np.empty(len(groups_this), dtype=np.float64)
                for j, group in enumerate(groups_this):
                    group_vect[j] = operation(params[group, i])
                result.append(group_vect)

    if len(result) == 0:
        return np.empty((0,))
    return np.concatenate(result)


def vect_to_params(vect, params, modes, groups=None):
    """Convert a vector from least squares optimization to an
    array of per-feature parameters.

    Parameters
    ----------
    vect : 1d ndarray of floats
        Vector containing the parameters for leastsq optimization.
    params : 2d ndarray of floats
        The array of per-feature parameters. Only the parameters that are not
        being optimized (the 'constant' parameters) are used, the others are
        replaced (not inplace).
    modes: ndarray of integers
        modes of every variable in the array of per-feature parameters:
        - 0 corresponds to constant
        - 1 corresponds to varying
        - 2 corresponds to varying, but equal for each feature
        - 3 corresponds to varying, but equal within cluster
        others are custom (e.g. per particle, per frame)
    groups : iterable of lists of lists
        Nested lists of integers defining groups of parameters. ``groups[0]``
        corresponds to groups of indices defining the clusters. Other elements
        correspond to custom parameter modes (index = n_mode - 3). If the only
        existing modes are 0, 1, and 3, this parameter can be omitted, as
        minimization will be grouped into clusters anyway.

    Returns
    -------
    Array of per-feature parameters. Its shape equals the shape of the ``params``
    arg. On the axes are (<feature>, <parameter>). The parameter order is
    (background, signal, <pos>, <size>, <other>).

    See also
    --------
    refine_leastsq
    """
    n, n_vars = params.shape
    assert len(modes) == n_vars
    assert min(modes) >= 0

    result = params.copy()
    current = 0
    for i, mode in enumerate(modes):
        mode = int(mode)
        if mode == 0:
            continue  # skip: it is constant
        elif mode == 1:
            # take all
            result[:, i] = vect[current:current + n]
            current += n
        elif mode == 2 or groups is None:
            # take only one
            result[:, i] = vect[current]
            current += 1
        else:
            # take the first for every unique id
            try:
                groups_this = groups[mode - 3]
            except (IndexError, TypeError):
                raise ValueError("The groups for mode {} were not provided".format(mode))
            for group, value in zip(groups_this, vect[current:current + len(groups_this)]):
                result[group, i] = value
            current += len(groups_this)

    return result


class FitFunctions:
    """Helper class maintaining fit functions and bounds.

    See also
    --------
    refine_leastsq
    """
    def __init__(self, fit_function='gauss', ndim=2, isotropic=True,
                 param_mode=None):
        self.ndim = ndim
        self.isotropic = isotropic
        self.pos_columns = default_pos_columns(ndim)
        self.size_columns = default_size_columns(ndim, isotropic)

        self.has_jacobian = True
        if not isinstance(fit_function, dict):
            if fit_function in function_templates:
                fit_function = function_templates[fit_function]
            else:
                splitted = fit_function.split('_')
                fit_function = '_'.join(splitted[:-1])
                order = splitted[-1]
                if fit_function in function_templates:
                    fit_function = function_templates[fit_function].copy()
                    fit_function['params'] = ['signal_mult'] + ['param_' + chr(i) for i in range(97, 97 + int(order))]
                    fit_function['default'] = {p: 1. for p in fit_function['params']}
                else:
                    ValueError("Unknown fit function {}".format(fit_function))

        self._params = list(fit_function['params'])
        self.fun = fit_function['fun']
        self.dfun = fit_function.get('dfun', None)
        self.default = dict(background=0., **fit_function.get('default', dict()))
        self.continuous = fit_function.get('continuous', False)
        self.has_jacobian = self.dfun is not None
        self.params = ['background', 'signal'] + self.pos_columns + \
                      self.size_columns + self._params

        _default_param_mode = dict(signal='var', background='cluster')
        if param_mode is None:
            self.param_mode = _default_param_mode
        else:
            self.param_mode = dict(_default_param_mode, **param_mode)

        # Broadcast 'param_mode' to all pos_columns
        if 'pos' in self.param_mode:
            for col in self.pos_columns:
                if col not in self.param_mode:
                    self.param_mode[col] = self.param_mode['pos']
            del self.param_mode['pos']
        # Broadcast 'param_mode' to all size_columns
        if (not isotropic) and ('size' in self.param_mode):
            for col in self.size_columns:
                if col not in self.param_mode:
                    self.param_mode[col] = self.param_mode['size']
            del self.param_mode['size']

        # Replace all string values with integers
        for key in self.param_mode:
            self.param_mode[key] = MODE_DICT[self.param_mode[key]]

        # set default values for position
        for param in self.pos_columns:
            if param not in self.param_mode:
                self.param_mode[param] = 1
        # set default values for all others to const
        for param in self.params:
            if param not in self.param_mode:
                self.param_mode[param] = 0
                self.param_mode[param] = 0

        if self.param_mode['background'] == 1:
            warnings.warn('The background param mode cannot vary per feature. '
                          'Varying per cluster now.')
            self.param_mode['background'] = 3

        self.modes = [int(self.param_mode[p]) for p in self.params]

        if ndim == 2 and isotropic and self.continuous:
            self.r2_fun, self.dr2_fun = r2_isotropic_2d, dr2_isotropic_2d
        elif ndim == 2 and not isotropic and self.continuous:
            self.r2_fun, self.dr2_fun = r2_anisotropic_2d, dr2_anisotropic_2d
        elif ndim == 3 and isotropic and self.continuous:
            self.r2_fun, self.dr2_fun = r2_isotropic_3d, dr2_isotropic_3d
        elif ndim == 3 and not isotropic and self.continuous:
            self.r2_fun, self.dr2_fun = r2_anisotropic_3d, dr2_anisotropic_3d
        elif ndim == 2 and isotropic:
            self.r2_fun, self.dr2_fun = r2_isotropic_2d_safe, dr2_isotropic_2d
        elif ndim == 2 and not isotropic:
            self.r2_fun, self.dr2_fun = r2_anisotropic_2d_safe, dr2_anisotropic_2d
        elif ndim == 3 and isotropic:
            self.r2_fun, self.dr2_fun = r2_isotropic_3d_safe, dr2_isotropic_3d
        elif ndim == 3 and not isotropic:
            self.r2_fun, self.dr2_fun = r2_anisotropic_3d_safe, dr2_anisotropic_3d
        else:
            raise ValueError()

    def plot_single_radial(self, r, **params):
        p = [params[_name] for _name in self._params]
        signal = params.get('signal', 1.)
        background = params.get('background', 0.)
        return background + signal * self.fun(r**2, p, self.ndim)

    def get_residual(self, images, meshes, masks, params_const,
                     groups=None, norm=1.):
        n, n_vars = params_const.shape
        if groups is None:  # assume all features are in the same cluster
            cl_groups = [np.arange(n)]
        else:
            cl_groups = groups[0]
        r2_fun = self.r2_fun
        dr2_fun = self.dr2_fun
        model_fun = self.fun
        model_dfun = self.dfun
        n_fun_params = len(self._params)
        ndim = self.ndim
        modes = self.modes

        def residual(vect):
            if np.any(np.isnan(vect)):
                raise RefineException('Parameters contained NaN values.')
            params = vect_to_params(vect, params_const, modes, groups)
            result = 0.
            for indices, image, mesh, masks_cl in zip(cl_groups, images, meshes,
                                                      masks):
                background = params[indices[0], 0]
                diff = image - background
                for i, mask in zip(indices, masks_cl):
                    r2 = r2_fun(mesh[:, mask], params[i])
                    signal = params[i, 1]
                    diff[mask] -= signal * model_fun(r2, params[i, -n_fun_params:], ndim)
                result += np.nansum(diff**2) / len(image)  # residual is per pixel
            return result / norm

        if not self.has_jacobian:
            return residual, None

        def jacobian(vect):
            if np.any(np.isnan(vect)):
                raise RefineException('Parameters contained NaN values.')
            params = vect_to_params(vect, params_const, modes, groups)
            result = params.copy()
            for indices, image, mesh, masks_cl in zip(cl_groups, images, meshes,
                                                      masks):
                n_cluster = len(indices)
                background = params[indices[0], 0]
                diff = image - background
                # array containing all derivatives
                derivs = np.zeros((n_cluster, n_vars - 1, len(image)))
                for j, (i, mask) in enumerate(zip(indices, masks_cl)):
                    r2 = r2_fun(mesh[:, mask], params[i])
                    dr2dx = dr2_fun(mesh[:, mask], params[i])
                    signal = params[i, 1]
                    model, deriv = model_dfun(r2, params[i, -n_fun_params:], ndim)
                    assert len(deriv) == n_fun_params + 1
                    diff[mask] -= signal * model
                    # model derivative wrt signal
                    derivs[j, 0, mask] = model
                    # evaluate model derivs wrt centers/sizes with chain rule
                    # numpy apparently transposes the left array ??
                    derivs[j, 1:1 + len(dr2dx), mask] = signal * (deriv[0] * dr2dx).T
                    # other derivatives
                    if n_fun_params > 0:
                        derivs[j, -n_fun_params:, mask] = signal * np.array(deriv[1:]).T
                # residual is per pixel
                result[indices, 1:] = np.nansum(-2 * diff * derivs, axis=2) / len(image)
                # background derivatives will be summed, so divide by n_cluster
                result[indices, 0] = np.nansum(-2 * diff) / (n_cluster * len(image))

            return vect_from_params(result, modes, groups, operation=np.sum) / norm

        return residual, jacobian


    def validate_bounds(self, bounds=None, radius=None):
        if bounds is None:
            bounds = dict()
        abs_arr = np.empty((2, len(self.params)), dtype=np.float64)
        diff_arr = np.empty((2, len(self.params)), dtype=np.float64)
        reldiff_arr = np.empty((2, len(self.params)), dtype=np.float64)

        for i, param in enumerate(self.params):
            abs_bnd = bounds.get(param, np.nan)
            diff = bounds.get(param + '_abs', np.nan)
            reldiff = bounds.get(param + '_rel', np.nan)

            # do some broadcasting if necessary
            if abs_bnd is np.nan and param in self.pos_columns:
                abs_bnd = bounds.get('pos', np.nan)
            if diff is np.nan and param in self.pos_columns:
                diff = bounds.get('pos_abs', np.nan)
            if reldiff is np.nan and param in self.pos_columns:
                reldiff = bounds.get('pos_rel', np.nan)
            if abs_bnd is np.nan and param in self.size_columns:
                abs_bnd = bounds.get('size', np.nan)
            if diff is np.nan and param in self.size_columns:
                diff = bounds.get('size_abs', np.nan)
            if reldiff is np.nan and param in self.size_columns:
                reldiff = bounds.get('size_rel', np.nan)

            if abs_bnd is np.nan:
                if param in ['background', 'signal'] + self.size_columns:
                    # by default, limit background, signal, and size to positive values
                    abs_bnd = (1E-7, np.nan)

            if diff is np.nan:
                if param in self.pos_columns:
                    # by default, limit position shifts to the mask radius
                    bnd = float(radius[self.pos_columns.index(param)])
                    diff = (bnd, bnd)

            abs_arr[:, i] = abs_bnd
            diff_arr[:, i] = diff
            reldiff_arr[:, i] = reldiff

        return abs_arr, diff_arr, reldiff_arr

    def compute_bounds(self, bounds, params, groups=None):
        abs, diff, reldiff = bounds

        with np.errstate(invalid='ignore'):  # for smooth comparison with np.nan
            with warnings.catch_warnings():  # for nanmax of only-NaN slice
                # compute the bounds: take the smallest bound possible
                warnings.simplefilter("ignore", RuntimeWarning)
                bound_low = np.nanmax([params - diff[0],      # abs. diff. bound
                                       params / reldiff[0]],  # rel. diff. bound
                                      axis=0)
                # do the absolute bound seperately for proper array broadcasting
                bound_low = np.fmax(bound_low, abs[0])
                bound_low[np.isnan(bound_low)] = -np.inf
                bound_high = np.nanmin([params + diff[1],
                                        params * reldiff[1]], axis=0)
                bound_high = np.fmin(bound_high, abs[1])
                bound_high[np.isnan(bound_high)] = np.inf
        # transform to vector so that it aligns with the parameter vector
        # when parameters are concatenated into one value, take the bound
        # as broad as possible (using min and max operations)
        bound_low = vect_from_params(bound_low, self.modes, groups,
                                     operation=np.min)
        bound_high = vect_from_params(bound_high, self.modes, groups,
                                      operation=np.max)
        return np.array([bound_low, bound_high], dtype=np.float64).T


def prepare_subimage(coords, image, radius):
    ndim = image.ndim
    radius = validate_tuple(radius, ndim)
    # slice region around cluster
    im, origin = slice_image(coords, image, radius)
    if origin is None:   # coordinates are out of image bounds
        raise RefineException('Coordinates are out of image bounds.')

    # include the edges where dist == 1 exactly
    dist = [(np.sum(((np.indices(im.shape).T - (coord - origin)) / radius)**2, -1) <= 1)
            for coord in coords]

    # to mask the image
    mask_total = np.any(dist, axis=0).T
    # to mask the masked image
    masks_singles = np.empty((len(coords), mask_total.sum()), dtype=bool)
    for i, _dist in enumerate(dist):
        masks_singles[i] = _dist.T[mask_total]

    # create the coordinates
    mesh = np.indices(im.shape, dtype=np.float64)[:, mask_total]
    # translate so that coordinates are in image coordinates
    mesh += np.array(origin)[:, np.newaxis]

    return im[mask_total].astype(np.float64), mesh, masks_singles


def prepare_subimages(coords, groups, frame_nos, reader, radius):
    # fast shortcut
    if groups is None:
        image, mesh, mask = prepare_subimage(coords, reader[frame_nos[0]],
                                             radius)
        return [image], [mesh], [mask]

    images = []
    meshes = []
    masks = []
    for cl_inds in groups[0]:
        frame_no = frame_nos[cl_inds[0]]
        image, mesh, mask = prepare_subimage(coords[cl_inds], reader[frame_no],
                                             radius)
        images.append(image)
        meshes.append(mesh)
        masks.append(mask)
    return images, meshes, masks


def refine_leastsq(f, reader, diameter, separation=None, fit_function='gauss',
                   param_mode=None, param_val=None, constraints=None,
                   bounds=None, compute_error=False, pos_columns=None,
                   t_column='frame', max_iter=10, max_shift=1, max_rms_dev=1.,
                   residual_factor=100000., **kwargs):
    """Refines overlapping feature coordinates by least-squares fitting to
    radial model functions.

    This does not raise an error if minimization fails. Instead, coordinates
    are unchanged and the added column ``cost`` will contain ``NaN``.

    Parameters
    ----------
    f : DataFrame
        pandas DataFrame containing coordinates of features.
        Required columns are the position columns (see ``pos_columns``)

        Any fit parameter (which are at least 'background', 'signal' and 'size')
        that is not present should be either given as a standard value in the
        ``param_val`` argument, or be present as a ``default`` value in the used
        fit function.

        If a FramesSequence is supplied as a reader, the time column (see
        ``t_column`` is also required.
    reader : pims.FramesSequence, pims.Frame, or ndarray
        A pims.FrameSequence is an object that returns an image when indexed. It
        also provides the ``frame_shape`` attribute. If not a FrameSequence is
        given a single image is assumed and all features that are present in
        ``f`` are assumed to be in that image.
    diameter : number or tuple
        Determines the feature mask diameter that is used for the refinement.
        Use a tuple to account for anisotropic pixel sizes (e.g. ``(7, 11)``).
    separation : number or tuple, optional
        Determines the distance below which features are considered in the same
        cluster. By default, equals ``diameter``. As the model feature function
        is only defined up to ``diameter``, it does not effect the refine
        results if this value is increased above ``diameter``.
    fit_function : string or or dict, optional
        The type of fit function. Either one of the default functions
        ``{'gauss', 'hat', 'ring', 'inv_series_<number>'}``, or a custom
        function defined using a dictionary. Defaults to ``'gauss'``.

        The fit function is loosely defined as follows:

        .. math::

            F(r, A, \\sigma, \\vec{p}) = B + A f(r, \\vec{p})

            r^2 = \\frac{x - c_x}{\\sigma_x}^2 + \\frac{y - c_y}{\\sigma_y}^2

        In which :math:`r` denotes the reduced distance to the feature center,
        :math:`B` the background intensity of the image, :math:`A` ('signal')
        the maximum value of the feature, :math:`\\vec{p}` a list of extra model
        parameters, :math:`\\sigma` ('size') the radial distance from the feature
        center at which the value of :math:`f(r)` has decayed to
        :math:`1/e \\approx 0.37`, and :math:`\\vec{c}` the coordinate of the
        feature center.

        So ``size`` is smaller than the apparent radius of the feature.
        Typically, it is three to four times lower than the ``diameter``.

        - The ``'gauss'`` function is a Gaussian, without any extra parameter
        - The ``'hat'`` function is solid disc of relative size ``disc_size``,
          and gaussian smoothed borders.
        - The ``'ring'`` model function is a displaced gaussian with parameter
          ``thickness``.
        - The ``inv_series_<number>`` model function is the inverse of an
          even polynomial containing ``<number>`` parameters
          (signal_mult / (1 + a r**2 + b r**4 + c r*2 + ...) ``signal_mult`` is
          best chosen such that the maximum of the polynomial equals 1.

        Define your own model function with a dictionary, containing:

        - params : list of str
          List of custom parameter names. The list has the same length as
          the ``p`` ndarray in ``fun`` and ``dfun``. It does not include
          ``background``, ``signal``, or ``size``.
        - fun : callable
          The image model function. It takes arguments ``(r2, p, ndim)``.

          - ``r2`` is a 1d ndarray containing the squared reduced
             radial distances (see the above definition).
          - ``p`` is an array of extra feature parameters
          - ``ndim`` is the number of dimensions in the image

          The function returns an ndarray of the same shape as ``r2``,
          containing the feature intensity values up to a maximum of 1.
        - dfun : callable, optional
          The analytical derivative of the image model function ``fun``.
          The function takes the same arguments as ``fun``.

          It returns a length-two tuple, with the following elements:

          1. (because of performance considerations)
             the image model function, exactly as returned by ``fun``
          2. the partial derivatives of ``fun`` in each point ``r2`` as a
             list of 1d ndarrays. The first element is the derivative with
             respect to ``r2``, the following elements w.r.t. the custom
             shape parameters as defined in ``params``. Hence, the number
             of elements in this list is ``len(params) + 1``.

        - default : dict, optional
          Default parameter values. For instance ``dict(thickness=0.2)``
        - continuous : boolean, optional
          Default True. Set this to False if :math:`f(|r|)` is not
          continuous at :math:`r = 0`. In that case, all pixels closer
          than 1 pixel to the center will be ignored.

    param_mode : dict, optional
        For each parameter, define whether it should be optimized or be
        kept constant. This also allows for constraining parameters to be equal
        within each cluster or equal for all frames.

        Each parameter can have one of the following values:

        * ``'var'`` : the parameter is allowed to vary for each feature independently
        * ``'const'`` : the parameter is not allowed to vary
        * ``'cluster'`` : the parameter is allowed to vary, but is equal
          within each cluster
        * ``'global'`` : the parameter is allowed to vary, but is equal for
          each feature
        * ``'frame'`` : Not yet implemented
        * ``'particle'`` : Not yet implemented

        Default values for position coordinates and signal is ``'var'``, for
        background ``'cluster'`` and for all others ``'const'``. Background
        cannot vary within one cluster, as regions overlap.
    param_val : dict, optional
        Default parameter values.
    constraints : tuple of dicts, optional
        Provide constraints for the parameters that are optimized. Each
        constraint consists of a dictionary containing the following elements:

        * type : str
          Constraint type: 'eq' for equality, which means that the constraint
          function result is to be zero. 'ineq' for inequality, which means
          that the constraint function result is to be greater than zero.
        * fun : callable
          The function defining the constraint. The function is provided
          a 3d ndarray with on the axes (<cluster>, <feature>, <parameter>)
          parameters are (background, signal, <pos>, <size>, <other>).
        * args : sequence, optional
          Extra arguments to be passed to the function.
        * cluster_size : integer
          Size of the cluster to which the constraint applies

        The parameter array that is presented to the constraint function is
        slightly different from the 2D array of per-feature parameters used in
        ``vect_from_params``, in the sense that that the first axis (axis 0) is
        extra.

        The 3D array of feature parameters that is presented to the constraint
        function is defined as follows:

        - Axis 0, the grouping axis, which  mostly has a length of 1, but in the
          case that the features that are optimized at once belong to different
          clusters (e.g. 1 image with 10 dimers) the length of this axis is the
          number of clusters that are optimized together (in this example, 10).
        - Axis 1, the feature axis, contains the individual features. In the example
          of 10 dimers, this axis would have a size of 2.
        - Axis 2, the parameter axis, contains the parameters. The order is
          ``['background', 'signal', <pos>, <size>, <extra>]``

    bounds: dict
        Bounds on parameters, in the following forms:

        - Absolute bounds ``{'x': [low, high]}``
        - Difference bounds, one-sided ``{'x_abs': max_diff}``
        - Difference bounds, two-sided ``{'x_abs': [max_diff_below, max_diff_above]}``
        - Relative bounds, one-sided ``{'x_rel': max_fraction_below}``
        - Relative bounds, two-sided ``{'x_rel': [max_fraction_below, max_fraction_above]}``

        When the keyword `pos` is used, this will be distributed to all
        pos_columns (but direct values of each positions will have precedence)
        When the keyword `size` is used, this will be distributed to all sizes,
        in the case of anisotropic sizes (also, direct values have precedence)

        For example, ``{'x': (2, 6), 'x_abs': (4, 6), 'x_rel': (1.5, 2.5)``
        would limit the parameter ``'x'`` between 2 and 6, between ``x-4`` and
        ``x+6``,  and between ``x/1.5`` and ``x*2.5``. The narrowest bound is
        taken.
    compute_error : boolean, optional
        Requires numdifftools to be installed. Default False.
        This is an experimental and untested feature that estimates the error
        in the optimized parameters on a per-feature basis from the curvature
        (diagonal elements of the Hessian) of the objective function in the
        optimized point.
    pos_columns: list of strings, optional
        Column names that contain the position coordinates.
        Defaults to ``['y', 'x']`` or ``['z', 'y', 'x']``, if ``'z'`` exists.
    t_column: string, optional
        Column name that denotes the frame index. Default ``'frame'``.
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
    kwargs : optional
        other arguments are passed directly to scipy.minimize. Defaults are
        ``dict(method='SLSQP', tol=1E-6, options=dict(maxiter=100, disp=False))``


    Returns
    -------
    DataFrame of refined coordinates. Added columns:

    * 'cluster': the cluster id of the feature.
    * 'cluster_size': the size of the cluster to which the feature belongs
    * 'cost': root mean squared difference between the final fit and
      the (preprocessed) image, in units of the cluster maximum value. If the
      optimization fails, no error is raised feature fields are unchanged,
      and this field becomes NaN.
    * (experimental) standard errors of variable parameters ('x_std', etc.) (only if compute_error is true)

    See also
    --------
    FitFunctions, vect_from_params, vect_to_params, wrap_constraint

    Notes
    -----
    This feature is a recent addition to trackpy that is still in its
    experimental phase. Please report any issues you encounter on Github.

    If you use this specific algorithm for your scientific publications, please
    mention the accompanying publication [1]_

    References
    ----------
    .. [1] van der Wel C., Kraft D.J. Automated tracking of colloidal clusters
    with sub-pixel accuracy and precision. J. Phys. Condens. Mat. 29:44001 (2017)
    DOI: http://dx.doi.org/10.1088/1361-648X/29/4/044001
    """
    if is_scipy_15:
        # see https://github.com/scipy/scipy/pull/13009
        warnings.warn(
            "refine_leastsq does not work well with scipy 1.5.*. "
            "We recommend upgrading or downgrading the scipy version."
        )

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
            raise ValueError('The provided reader neither has the attribute '
                             '"frame_shape" nor the attribute "ndim". Please '
                             'provide a pims.FramesSequence (for refinement of '
                             'multiple frames) or a pims.Frame / ndarray (for '
                             'refinement of a single frame).')
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

    if ndim != len(pos_columns):
        raise ValueError('The image dimensionality ({}) does not match the '
                         'number of dimensions in the feature DataFrame ({})'
                         ''.format(ndim, str(pos_columns)))
    if t_column not in f:
        raise ValueError('The expected column for frame indices ("{0}") is not '
                         'present in the supplied feature DataFrame. This '
                         'is required when refining a FramesSequence. Either '
                         'add the "{0}" column or change the "t_column" '
                         'argument.'.format(t_column))

    diameter = validate_tuple(diameter, ndim)
    radius = tuple([x//2 for x in diameter])
    isotropic = is_isotropic(diameter)
    if separation is None:
        separation = diameter

    ff = FitFunctions(fit_function, ndim, isotropic, param_mode)

    if constraints is None:
        constraints = dict()

    # makes a copy
    f = cluster(f, separation, pos_columns, t_column)

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
        raise NotImplemented("Currently, parameters can only be optimized "
                             "on a per-feature, per-cluster, or global basis. "
                             "Please feel free to implement per-frame or "
                             "per-trajectory optimization!")

    last_frame = None  # just for logging
    for _, f_iter in iterable:
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
                raise RefineException('Not all initial parameters ({}) are known.'.format(ff.params))
            # extract the coordinates from the parameter array
            coords = params[:, 2:2+ndim]
            # transform the params into a vector for leastq optimization
            vect = vect_from_params(params, ff.modes, groups, operation=np.mean)

            f_constraints = _wrap_constraints(constraints, params, ff.modes,
                                              groups)
            f_bounds = ff.compute_bounds(bounds, params, groups)
            for _n_iter in range(max_iter):
                sub_images, meshes, masks = prepare_subimages(coords, groups,
                                                              frame_nos, frames,
                                                              radius)
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
                raise RefineException('The rms deviation of the fit ({:.4f} is'
                                      'more than the maximum value of '
                                      '{:.4f}.'.format(rms_dev, max_rms_dev))

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
            logger.warn('RefineException: {}'.format(e.args))
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
    """Obtain fit parameters from an image of well-separated features with known
    location, in order to be able to use them in ``refine_leastsq``.
    The locations are first optimized by center of mass, and then the shape
    parameters are optimized while keeping the locations fixed.

    This function is still experimental and untested."""
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
    size_columns = default_size_columns(ndim, isotropic)

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


# The following functions define the default fit functions


def r2_isotropic_2d(mesh, p):
    y, x = mesh
    cy, cx, size = p[2:5]
    return ((x-cx)**2 + (y-cy)**2) / size**2


def r2_isotropic_2d_safe(mesh, p):
    y, x = mesh
    cy, cx, size = p[2:5]
    dist = (x-cx)**2 + (y-cy)**2
    dist[dist < 1.] = np.nan
    dist /= size**2
    return dist


def dr2_isotropic_2d(mesh, p):
    y, x = mesh
    cy, cx, size = p[2:5]
    return np.vstack([(cy-y) * (2. / size**2),
                      (cx-x) * (2. / size**2),
                      ((x-cx)**2 + (y-cy)**2) * (-2. / size**3)])


def r2_isotropic_3d(mesh, p):
    z, y, x = mesh
    cz, cy, cx, size = p[2:6]
    return ((x-cx)**2 + (y-cy)**2 + (z-cz)**2) / size**2


def r2_isotropic_3d_safe(mesh, p):
    z, y, x = mesh
    cz, cy, cx, size = p[2:6]
    dist = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
    dist[dist < 1.] = np.nan
    dist /= size**2
    return dist


def dr2_isotropic_3d(mesh, p):
    z, y, x = mesh
    cz, cy, cx, size = p[2:6]
    return np.vstack([(cz-z) * (2. / size**2),
                      (cy-y) * (2. / size**2),
                      (cx-x) * (2. / size**2),
                      ((x-cx)**2 + (y-cy)**2 + (z-cz)**2) * (-2. / size**3)])


def r2_anisotropic_2d(mesh, p):
    y, x = mesh
    cy, cx, size_y, size_x = p[2:6]
    return (x-cx)**2 / size_x**2 + (y-cy)**2 / size_y**2


def r2_anisotropic_2d_safe(mesh, p):
    y, x = mesh
    cy, cx, size_y, size_x = p[2:6]
    mask = (x-cx)**2 + (y-cy)**2 < 1.
    result = (x-cx)**2 / size_x**2 + (y-cy)**2 / size_y**2
    result[mask] = np.nan
    return result


def dr2_anisotropic_2d(mesh, p):
    y, x = mesh
    cy, cx, size_y, size_x = p[2:6]
    return np.vstack([(cy-y) * (2. / size_y**2),
                      (cx-x) * (2. / size_x**2),
                      (y-cy)**2 * (-2. / size_y**3),
                      (x-cx)**2 * (-2. / size_x**3)])


def r2_anisotropic_3d(mesh, p):
    z, y, x = mesh
    cz, cy, cx, size_z, size_y, size_x = p[2:8]
    return (x-cx)**2 / size_x**2 + (y-cy)**2 / size_y**2 + (z-cz)**2 / size_z**2


def r2_anisotropic_3d_safe(mesh, p):
    z, y, x = mesh
    cz, cy, cx, size_z, size_y, size_x = p[2:8]
    mask = (x-cx)**2 + (y-cy)**2 + (z-cz)**2 < 1.
    result = (x-cx)**2 / size_x**2 + (y-cy)**2 / size_y**2 + \
             (z-cz)**2 / size_z**2
    result[mask] = np.nan
    return result


def dr2_anisotropic_3d(mesh, p):
    z, y, x = mesh
    cz, cy, cx, size_z, size_y, size_x = p[2:8]
    return np.vstack([(cz-z) * (2. / size_z**2),
                      (cy-y) * (2. / size_y**2),
                      (cx-x) * (2. / size_x**2),
                      (z-cz)**2 * (-2. / size_z**3),
                      (y-cy)**2 * (-2. / size_y**3),
                      (x-cx)**2 * (-2. / size_x**3)])


def gauss_fun(r2, p, ndim):
    return safe_exp(-0.5*ndim*r2)


def gauss_dfun(r2, p, ndim):
    func = safe_exp(-0.5*ndim*r2)
    return func, [-0.5*ndim*func]


def disc_fun(r2, p, ndim):
    result = np.ones_like(r2)
    disc_size = p[0]
    if disc_size <= 0:
        return gauss_fun(r2, None, ndim)
    elif disc_size >= 1.:
        disc_size = 0.999
    mask = r2 > disc_size**2
    result[mask] = safe_exp(((r2[mask]**0.5 - disc_size)/(1 - disc_size))**2 *
                            ndim/-2)
    return result


def ring_fun(r2, p, ndim):
    t = p[0]
    r = r2**0.5
    return safe_exp(-0.5 * ndim * ((r - 1 + t)/t)**2)


def ring_dfun(r2, p, ndim):
    t = p[0]
    r = r2**0.5
    num = r - 1 + t
    func = safe_exp(-0.5 * ndim * (num/t)**2)
    return func, [func * (-0.5*ndim / (r*t**2)) * num,
                  func * ndim * (num**2/t**3 - num / t**2)]


def inv_series_fun(r2, p, ndim):
    """ p is a vector of arguments [mult, a, b, c, ...], defining the series:
    signal_mult / (1 + a r^2 + b r^4 + c r^6 + ...)
    """
    series_param = np.array(p)
    series_param[0] = 1.
    return p[0] / np.polyval(series_param, r2)


# TODO: fix the derivative of the inv_series
# def inv_series_dfun(r2, p, ndim):
#     """ p is a vector of arguments [mult, a, b, c, ...], defining the series:
#     signal_mult / (1 + a r^2 + b r^4 + c r^6 + ...)
#     """
#     # n = np.arange(1, len(p))
#     # series_param = np.ones_like(p, dtype=np.float64)
#     # prefactors = (ndim / 2)**n / np.cumprod(n)
#     # series_param[1:] = p[1:] * prefactors
#     # func = np.polyval(series_param, r2)
#     #
#     # dseries_param = p[1:] * prefactors * n
#     # dr2 = -p[0] * func**-2 * np.polyval(dseries_param, r2)
#     # dmult = 1 / func
#     # dparam = -p[0] * func**-2 * (prefactors[:, np.newaxis] * r2**n[:, np.newaxis])
#     n = np.arange(1, len(p))
#     series_param = np.ones_like(p, dtype=np.float64)
#     series_param[1:] = p[1:] * (ndim / 2)**n / np.cumprod(n)
#     func = 1 / np.polyval(series_param, r2)
#
#     dr2 = -p[0] * func**2 * np.polyval(series_param[1:] * n, r2)
#     dmult = func
#     prefactors = series_param[1:] / p[1:]
#     dparam = -p[0] * func**-2 * (prefactors[:, np.newaxis] * r2**n[:, np.newaxis])
#
#     return p[0] * func, np.vstack([[dr2, dmult], dparam])


function_templates = dict(gauss=dict(params=[], fun=gauss_fun,
                                     dfun=gauss_dfun,
                                     continuous=True),
                          ring=dict(params=['thickness'], fun=ring_fun,
                                    dfun=ring_dfun,
                                    default=dict(thickness=0.2),
                                    continuous=False),
                          disc=dict(params=['disc_size'], fun=disc_fun,
                                    default=dict(disc_size=0.5),
                                    continuous=True),
                          inv_series=dict(fun=inv_series_fun,
                                          continuous=True))


# The following functions manage the constraints to be used in refine_leastsq


def _wrap_constraints(constraints, params_const, modes, groups=None):
    """Wraps a list of constraints such that their functions do not have to
    interpret the vector of parameters, but an array of per-feature parameters
    instead.

    The parameter array that is presented to the constraint function is slightly
    different from the 2D array of per-feature parameters used in
    ``vect_from_params``, in the sense that that the first axis (axis 0) is
    extra.

    The 3D array of feature parameters that is presented to the constraint
    function is defined as follows:
    - Axis 0, the grouping axis, which  mostly has a length of 1, but in the
    case that the features that are optimized at once belong to different
    clusters (e.g. 1 image with 10 dimers) the length of this axis is the
    number of clusters that are optimized together (in this example, 10).
    - Axis 1, the feature axis, contains the individual features. In the example
    of 10 dimers, this axis would have a size of 2.
    - Axis 2, the parameter axis, contains the parameters. The order is
    ``['background', 'signal', <pos>, <size>, <extra>]``

    Parameters
    ----------
    constraints : iterable of dicts
        Contains definition
        These are described as follows (adapted from scipy.optimize.minimize):

            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint. The function is provided
                a 3d ndarray with on the axes (<cluster>, <feature>, <parameter>)
                parameters are (background, signal, <pos>, <size>, <other>)
            args : sequence, optional
                Extra arguments to be passed to the function.
            cluster_size : integer
                Size of the cluster to which the constraint applies

        Equality constraint means that the constraint function result is to
        be zero whereas inequality means that it is to be non-negative.
    params_const : ndarray
        The array of per-feature parameters. Only the parameters that are not
        being optimized (the 'constant' parameters) are used, the others are
        replaced (not inplace).
    modes: ndarray of integers
        modes of every variable in the array of per-feature parameters:
        - 0 corresponds to constant
        - 1 corresponds to varying
        - 2 corresponds to varying, but equal for each feature
        - 3 corresponds to varying, but equal within cluster
        others are custom (e.g. per particle, per frame)
    groups : iterable of lists
        list of integers defining groups of parameters. The first element
        correspond to the indices defining clusters: other elements correspond
        to custom parameter modes (index = n_mode - 3). If the only existing
        modes are 0, 1, and 3, this parameter can be left out, as
        minimization will be grouped into clusters anyway.

    See also
    --------
    dimer, trimer, tetramer
    """
    if constraints is None:
        return []

    if groups is not None:
        cl_sizes = np.array([len(g) for g in groups[0]], dtype=int)

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
                raise ValueError('Please provide the groups argument when using'
                                 ' a per-group parameter at the same time as '
                                 'optimizing multiple groups at once.')
            # provide all parameters to the constraint
            def wrapped(vect, *args, **kwargs):
                params = vect_to_params(vect, params_const, modes, groups)
                return cons['fun'](params[np.newaxis, :, :], *args, **kwargs)
        elif cluster_size in cl_sizes:
            groups_this = groups[0][cl_sizes == cluster_size]
            if len(groups_this) == 0:
                continue  # there are no clusters of the constraint size
            # group the appropriate clusters together and return multiple values
            def wrapped(vect, *args, **kwargs):
                params = vect_to_params(vect, params_const, modes, groups)
                params_grouped = np.array([params[g] for g in groups_this])
                return cons['fun'](params_grouped, *args, **kwargs)
        else:
            raise ValueError('The following constraint could not be '
                             'wrapped: ' + str(cons))
        cons_wrapped = cons.copy()
        cons_wrapped['fun'] = wrapped
        result.append(cons_wrapped)
        if 'jac' in cons_wrapped:
            logger.warn('Constraint jacobians are not implemented')
            del cons_wrapped['jac']
    return result


def _dimer_fun(x, dist, ndim):
    pos = x[..., 2:2+ndim]  # get positions only
    return 1 - np.sum(((pos[:, 0] - pos[:, 1])/dist)**2, axis=1)


def dimer(dist, ndim=2):
    """Constraint setting clusters of 2 at a fixed distance.

    Provide a tuple as distance to deal with anisotropic pixel sizes."""
    dist = np.array(validate_tuple(dist, ndim))
    return (dict(type='eq', cluster_size=2, fun=_dimer_fun, args=(dist, ndim)),)


def _trimer_fun(x, dist, ndim):
    x = x[..., 2:2+ndim]  # get positions only
    return np.concatenate((1 - np.sum(((x[:, 0] - x[:, 1])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 1] - x[:, 2])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 0] - x[:, 2])/dist)**2, axis=1)))


def trimer(dist, ndim=2):
    """Constraint setting clusters of 3 at a fixed distance from each other.

    This contains 3 constraints. Provide a tuple as distance to deal with
    anisotropic pixel sizes."""
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
    # take the four smallest and do not test the other two: they are fixed by
    # the four first constraints.
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
    """Constraint setting clusters of 4 at a fixed distance from each other.

    For 2D: features are in a perfect square (4 constraints)
    For 3D: features are constrained in a tetrahedron (6 constraints).
    Provide a tuple as distance to deal with
    anisotropic pixel sizes."""
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

    Allows pixel anisotropy by providing ``mpp``, microns per pixel. The
    number of constraints equals the number of frames - 1."""
    mpp = np.array(validate_tuple(mpp, ndim))
    return (dict(type='eq', fun=_dimer_fun_global, args=(mpp, ndim,)),)
