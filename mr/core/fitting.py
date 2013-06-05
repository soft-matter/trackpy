# Copyright 2012 Daniel B. Allan
# dallan@pha.jhu.edu, daniel.b.allan@gmail.com
# http://pha.jhu.edu/~dallan
# http://www.danallan.com
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses>.

from __future__ import division
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import stats
from scipy.interpolate import UnivariateSpline
import lmfit
from lmfit import Parameters

class Result:
    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, value):
        self._values = value
    
    @property
    def stderr(self):
        return self._stderr

    @stderr.setter
    def stderr(self, value):
        self._stderr = value

    @property
    def residual(self, value):
        return self._residual

    @residual.setter
    def residual(self, value):
        self._residual = value

    @property
    def fits(self, value):
        return self._fits

    @fits.setter
    def fits(self, value):
        self._fits= value

def NLS(data, model_func, params, weights=None,
        log_residual=False, inverted_model=False, plot=True):
    """Perform a nonlinear least-sqaured fit on each column of a DataFrame. 

    Parameters
    ----------
    data : a DataFrame or Series indexed by the exogenous ("x") variable.
        Missing values will be ignored.
    model_func : model function of the form f(x, params)
    params : a Parameters object or a function of the form f(data) that returns
        a Parameters object. (See the lmfit module for more on Parameters.)
    weights : Series
    log_residual : boolean, default False
        Compute the residual in log space.
    inverted_model : boolean, default False
        Use when the model is expressed as x(y).
    plot : boolean, default True
        Automatically plot fits.

    Returns
    -------
    results : DataFrame with a column of best fit params for each 
        column of data.

    ------
    a Warning if the fit fails to converge

    Notes
    -----
    This wraps lmfit, which extends scipy.optimize.leastsq, which itself wraps 
    an old Fortran MINPACK implementation of the Levenburg-Marquardt algorithm. 
    """
    p = params # notational convenience
    pd.set_option('use_inf_as_null', True)
    def residual_func(params, x, y, weights):
        modeled_y = x.apply(lambda x: model_func(x, params))
        e = (y - modeled_y)
        e.fillna(e.mean(), inplace=True)
        if weights is None:
            return e.values
        else:
            return e.mul(weights).values
    def inverse_residual_func(params, x, y, weights):
        modeled_x = y.apply(lambda y: model_func(y, params)).dropna()
        spline = UnivariateSpline(modeled_x, y.reindex(modeled_x.index), k=3)
        modeled_y = spline(x)
        e = (y - modeled_y)
        e.fillna(e.mean(), inplace=True)
        if weights is None:
            return e.values
        else:
            return e.mul(weights).values
        
    x = Series(data.index.values, index=data.index, dtype=np.float64)
    if weights is not None:
        assert weights.size == x.size, \
            "weights must be an array-like sequence the same length as data."
        weights = Series(np.asarray(weights), index=x.index)
    values = DataFrame(index=p.keys())
    stderr = DataFrame(index=p.keys())
    y = data.dropna().astype('float64')
    if not inverted_model:
        result = lmfit.minimize(residual_func, p, args=(x, y, weights))
    else:
        result = lmfit.minimize(inverse_residual_func, p, args=(x, y, weights))
    result_params = Series(result.params)
    values = result_params.apply(lambda param: param.value)
    stderr = result_params.apply(lambda param: param.stderr)
    residual = Series(result.residual, index=x)
    pd.reset_option('use_inf_as_null')
    r = Result()
    r.values = values.T
    r.stderr = stderr.T
    r.residual = residual
    r.model = lambda x: model_func(x, result.params) # curried
    if plot:
        import plots
        plots.fit(data, r.fits, inverted_model)
    return r

def fit_powerlaw(data, plot=True):
    """Fit a powerlaw by doing a linear regression in log space."""
    ys = DataFrame(data)
    x = Series(data.index.values, index=data.index, dtype=np.float64)
    values = DataFrame(index=['n', 'A']) 
    fits = {}
    for col in ys:
        y = ys[col].dropna()
        slope, intercept, r, p, stderr = \
            stats.linregress(np.log(x), np.log(y))
        print 'slope', slope, ', intercept', intercept
        values[col] = [slope, np.exp(intercept)]
        fits[col] = x.apply(lambda x: np.exp(intercept)*x**slope) 
    values = values.T
    fits = pd.concat(fits, axis=1)
    if plot:
        import plots
        plots.fit(data, fits, logx=True, logy=True)
    return values

