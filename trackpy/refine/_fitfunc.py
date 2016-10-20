from __future__ import division, print_function, absolute_import

import six
from ..utils import default_pos_columns, default_size_columns, RefineException
import numpy as np
import warnings


MODE_DICT = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
             'const': 0, 'var': 1, 'global': 2, 'cluster': 3,
             'particle': 4, 'frame': 5}


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


def gauss_func(r2, p, ndim):
    return np.exp(-0.5*ndim*r2)


def gauss_dfunc(r2, p, ndim):
    func = np.exp(-0.5*ndim*r2)
    return func, [-0.5*ndim*func]


def disc_func(r2, p, ndim):
    result = np.ones_like(r2)
    disc_size = p[0]
    if disc_size <= 0:
        return gauss_func(r2, None, ndim)
    elif disc_size >= 1.:
        disc_size = 0.999
    mask = r2 > disc_size**2
    result[mask] = np.exp(((r2[mask]**0.5 - disc_size)/(1 - disc_size))**2 *
                          ndim/-2)
    return result


def ring_func(r2, p, ndim):
    t = p[0]
    r = r2**0.5
    return np.exp(-0.5 * ndim * ((r - 1 + t)/t)**2)


def ring_dfunc(r2, p, ndim):
    t = p[0]
    r = r2**0.5
    num = r - 1 + t
    func = np.exp(-0.5 * ndim * (num/t)**2)
    return func, [func * (-0.5*ndim / (r*t**2)) * num,
                  func * ndim * (num**2/t**3 - num / t**2)]


def inv_series_func(r2, p, ndim):
    """ p is a vector of arguments [mult, a, b, c, ...], defining the series:
    signal_mult / (1 + a r^2 + b r^4 + c r^6 + ...)
    """
    series_param = np.array(p)
    series_param[0] = 1.
    return p[0] / np.polyval(series_param, r2)


## def inv_series_func(r2, p, ndim):
#     """ p is a vector of arguments [mult, a, b, c, ...], defining the series:
#     signal_mult / (1 + a r^2 + b r^4 + c r^6 + ...)
#     """
#     n = np.arange(1, len(p))
#     series_param = np.ones_like(p, dtype=np.float64)
#     series_param[1:] = p[1:] * (ndim / 2)**n / np.cumprod(n)
#     return p[0] / np.polyval(series_param, r2)

# def inv_series_dfunc(r2, p, ndim):
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



function_templates = dict(gauss=dict(params=[], func=gauss_func,
                                     dfunc=gauss_dfunc,
                                     continuous=True),
                          ring=dict(params=['thickness'], func=ring_func,
                                    dfunc=ring_dfunc,
                                    default=dict(thickness=0.5)),
                          disc=dict(params=['disc_size'], func=disc_func,
                                    default=dict(disc_size=0.5)),
                          inv_series=dict(func=inv_series_func,
                                          continuous=True))


def vect_from_params(params, modes, groups=None, operation=None):
    """Convert an array of per-feature parameters into a vector
    for leastsquares optimization

    Parameters
    ----------
    params : 2d ndarray of floats
    modes : 1d ndarray of integers
        modes of every variable along the first axis of var_array
        0 corresponds to constant
        1 corresponds to varying
        2 corresponds to varying, but equal for each feature
        3 corresponds to varying, but equal within cluster
        others are custom (e.g. per particle, per frame)
    groups : iterable of lists
        each dictionary contains the grouped indices
        indices correspond to n_mode - 3. (e.g. column 0 has cluster indices)
    operation : function
        function that converts a 1d array of parameters into a scalar
        Default None: take the first one (assume they are all equal)
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
    """Convert a vector from leastsquares optimization to an
    array of per-feature parameters.

    Parameters
    ----------
    vect : 1d ndarray of floats
        the vector from leastsquares optimization
    params : 2d ndarray of floats
        the array of per-feature parameters, to be able to know the constants
    modes : 1d ndarray of integers
        modes of every variable along the first axis of var_array
        0 corresponds to constant
        1 corresponds to varying
        2 corresponds to varying, but equal for each feature
        3 corresponds to varying, but equal within cluster
        others are custom (e.g. per particle, per frame)
    groups : iterable of lists
        each dictionary contains the grouped indices
        indices correspond to n_mode - 3. (e.g. column 0 has cluster indices)
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


class FitFunctions(object):
    """Helper class maintaining fit functions and bounds.

    See also
    --------
    clustertracking.refine_leastsq
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
        self.func = fit_function['func']
        self.dfunc = fit_function.get('dfunc', None)
        self.default = dict(background=0., **fit_function.get('default', dict()))
        self.continuous = fit_function.get('continuous', False)
        self.has_jacobian = self.dfunc is not None
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
            self.r2_func, self.dr2_func = r2_isotropic_2d, dr2_isotropic_2d
        elif ndim == 2 and not isotropic and self.continuous:
            self.r2_func, self.dr2_func = r2_anisotropic_2d, dr2_anisotropic_2d
        elif ndim == 3 and isotropic and self.continuous:
            self.r2_func, self.dr2_func = r2_isotropic_3d, dr2_isotropic_3d
        elif ndim == 3 and not isotropic and self.continuous:
            self.r2_func, self.dr2_func = r2_anisotropic_3d, dr2_anisotropic_3d
        elif ndim == 2 and isotropic:
            self.r2_func, self.dr2_func = r2_isotropic_2d_safe, dr2_isotropic_2d
        elif ndim == 2 and not isotropic:
            self.r2_func, self.dr2_func = r2_anisotropic_2d_safe, dr2_anisotropic_2d
        elif ndim == 3 and isotropic:
            self.r2_func, self.dr2_func = r2_isotropic_3d_safe, dr2_isotropic_3d
        elif ndim == 3 and not isotropic:
            self.r2_func, self.dr2_func = r2_anisotropic_3d_safe, dr2_anisotropic_3d
        else:
            raise ValueError()

    def plot_single_radial(self, r, **params):
        p = [params[_name] for _name in self._params]
        signal = params.get('signal', 1.)
        background = params.get('background', 0.)
        return background + signal * self.func(r**2, p, self.ndim)

    def get_residual(self, images, meshes, masks, params_const,
                     groups=None, norm=1.):
        n, n_vars = params_const.shape
        if groups is None:  # assume all features are in the same cluster
            cl_groups = [np.arange(n)]
        else:
            cl_groups = groups[0]
        r2_func = self.r2_func
        dr2_func = self.dr2_func
        model_func = self.func
        model_dfunc = self.dfunc
        n_func_params = len(self._params)
        ndim = self.ndim
        modes = self.modes

        def residual(vect):
            if np.any(np.isnan(vect)):
                raise RefineException
            params = vect_to_params(vect, params_const, modes, groups)
            result = 0.
            for indices, image, mesh, masks_cl in zip(cl_groups, images, meshes,
                                                      masks):
                background = params[indices[0], 0]
                diff = image - background
                for i, mask in zip(indices, masks_cl):
                    r2 = r2_func(mesh[:, mask], params[i])
                    signal = params[i, 1]
                    diff[mask] -= signal * model_func(r2, params[i, -n_func_params:], ndim)
                result += np.nansum(diff**2) / len(image)  # residual is per pixel
            return result / norm

        if not self.has_jacobian:
            return residual, None

        def jacobian(vect):
            if np.any(np.isnan(vect)):
                raise RefineException
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
                    r2 = r2_func(mesh[:, mask], params[i])
                    dr2dx = dr2_func(mesh[:, mask], params[i])
                    signal = params[i, 1]
                    model, deriv = model_dfunc(r2, params[i, -n_func_params:], ndim)
                    assert len(deriv) == n_func_params + 1
                    diff[mask] -= signal * model
                    # model derivative wrt signal
                    derivs[j, 0, mask] = model
                    # evaluate model derivs wrt centers/sizes with chain rule
                    # numpy apparently transposes the left array ??
                    derivs[j, 1:1 + len(dr2dx), mask] = signal * (deriv[0] * dr2dx).T
                    # other derivatives
                    if n_func_params > 0:
                        derivs[j, -n_func_params:, mask] = signal * np.array(deriv[1:]).T
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
            diff = bounds.get(param + '_diff', np.nan)
            reldiff = bounds.get(param + '_rel_diff', np.nan)

            # do some broadcasting if necessary
            if abs_bnd is np.nan and param in self.pos_columns:
                abs_bnd = bounds.get('pos', np.nan)
            if diff is np.nan and param in self.pos_columns:
                diff = bounds.get('pos_diff', np.nan)
            if reldiff is np.nan and param in self.pos_columns:
                reldiff = bounds.get('pos_rel_diff', np.nan)
            if abs_bnd is np.nan and param in self.size_columns:
                abs_bnd = bounds.get('size', np.nan)
            if diff is np.nan and param in self.size_columns:
                diff = bounds.get('size_diff', np.nan)
            if reldiff is np.nan and param in self.size_columns:
                reldiff = bounds.get('size_rel_diff', np.nan)

            if abs_bnd is np.nan:
                if param in ['background', 'signal'] + self.size_columns:
                    # by default, limit background, signal, and size to positive values
                    abs_bnd = (0., np.nan)

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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # compute the bounds: take the smallest bound possible
            bound_low = np.nanmax([params - diff[0],            # difference bound
                                   params * (1 - reldiff[0])],  # rel. diff. bound
                                  axis=0)
            # do the absolute bound seperately for proper array broadcasting
            bound_low = np.fmax(bound_low, abs[0])
            bound_low[np.isnan(bound_low)] = -np.inf
            bound_high = np.nanmin([params + diff[1],
                                    params * (1 + reldiff[1])], axis=0)
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
