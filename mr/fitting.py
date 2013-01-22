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
    def fitlines(self, value):
        return self._fitlines

    @fitlines.setter
    def fitlines(self, value):
        self._fitlines = value

def NLS(data, model_func, params, log_residual=False, inverted_model=False):
    """Perform a nonlinear least-sqaured fit on each column of a DataFrame. 

    Parameters
    ----------
    data : a DataFrame or Series indexed by the exogenous ("x") variable.
        Missing values will be ignored.
    model_func : model function of the form f(x, params)
    params : a Parameters object or a function of the form f(data) that returns
        a Parameters object. (See the lmfit module for more on Parameters.)
    log_residual : boolean, default False
        Compute the residual in log space.
    inverted_model : boolean, default False
        Use when the model is expressed as x(y).

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
    def residual_func(params, x, y):
        f = x.apply(lambda x: model_func(x, params))
        if log_residual:
            e = (np.log(y) - np.log(f))
            e.fillna(e.mean(), inplace=True)
        else:
            e = y - f
        return e.values
    # If we are given a params-generating function, generate sample
    # params to index the results DataFrame. 
    pd.set_option('use_inf_as_null', True)
    ys = DataFrame(data) # in case it's a Series
    x = Series(data.index.values, index=data.index, dtype=np.float64)
    if hasattr(params, '__call__'):
        p = params(ys.icol(0))
    values = DataFrame(index=p.keys())
    stderr = DataFrame(index=p.keys())
    residuals = {}
    for col in ys:
        y = ys[col].dropna()
        # If need be, generate params using this column's data. 
        if hasattr(params, '__call__'):
            p = params(y)
        else:
            p = params
        if not inverted_model:
            result = lmfit.minimize(residual_func, p, args=(x, y))
        else:
            result = lmfit.minimize(residual_func, p, args=(y, x))
        result_params = Series(result.params)
        values[col] = result_params.apply(lambda param: param.value)
        stderr[col] = result_params.apply(lambda param: param.stderr)
        residuals[col] = Series(result.residual, index=x)
    residuals = pd.concat(residuals, axis=1)
    pd.reset_option('use_inf_as_null')
    r = Result()
    r.values = values.T
    r.stderr = stderr.T
    r.residuals = residuals
    if not inverted_model:
        r.fitlines = ys + residuals
    else:
        r.fitlines = residuals.add(x, axis=0)
    return r

def fit_powerlaw(data):
    """Fit a powerlaw by doing a linear regression in log space."""
    data = DataFrame(data)
    fits = []
    for col in data:
        slope, intercept, r, p, stderr = \
            stats.linregress(data.index.values, data[col].values)
        fits.append(Series([slope, np.exp(intercept)], index=['A', 'n']))
    return pd.concat(fits, axis=1, keys=data.columns)
