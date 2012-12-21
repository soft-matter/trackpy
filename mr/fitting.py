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
from scipy import optimize

def parse_output(output):
    "Parse the output from scipy.optimize.leastsq, and respond appropriately."
    x, cov_x, infodict, mesg, ier = output
    if not ier in [1, 2, 3, 4]:
        raise Exception, "A solution was not found. Message:\n%s" % mesg
    return x

def fit(data, func, guess_params, collective=False):
    """Perform a least squared fit on a DataFrame, either fitting the columns
    collectively with one best fit, or producing a best fit for each column
    individually.

    Parameters
    ----------
    data : a DataFrame or Series indexed by the exogenous ("x") variable.
    func : model function of the form f(x, *params)
    guess_params : a sequence of parameters to be optimized
    collective : boolean
        If True, find the parameters that fit all the columns together.
        If False, fit each column individually.

    Returns
    -------
    best_params : DataFrame with a column of best fit params for each 
        column of data. (If collective=True, this is just a Series.)
    """
    exog = Series(data.index, index=data.index, dtype=np.float64)
    if collective:
        def err(params):
            f = exog.apply(lambda x: func(x, *params))
            # Subtract the model from each column of data. Then sum the
            # columns.
            return (data - f).sum(1).fillna(0).values
        output = optimize.leastsq(err, guess_params, full_output=True)
        best_params = parse_output(output)
        # If the guess_params come as a Series, reuse the same Index.
        try:
            if not issubclass(index, pandas.core.index.Index):
                raise TypeError
            index = guess_params.index
        except:
            index = None
        return Series(best_params, index=index)
    else:
        # Make an empty DataFrame of the right length to hold the fits.
        try:
            if not issubclass(index, pandas.core.index.Index):
                raise TypeError
            index = guess_params.index
        except:
            index = np.arange(len(guess_params))
        fits = DataFrame(index=index)
        data = DataFrame(data)
        for col in data:
            def err(params):
                f = exog.apply(lambda x: func(x, *params))
                e = (data[col] - f).fillna(0).values
                return e
            output = optimize.leastsq(err, guess_params, full_output=True)
            best_params = parse_output(output)
            fits[col] = Series(best_params)
        return fits.T
