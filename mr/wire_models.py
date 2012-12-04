import numpy as np
from scipy import special
from scipy import optimize
import pandas as pd

class Model(object):
    "Nonlinear model"
    def __init__(self, endog, exog, weights=None, sigma=None):
        self.endog = endog
        self.exog = exog
        if weights is not None:
            self.weights = weights
        elif sigma:
            self.weights = 1./sigma
        else:
            self.weights = None

    def fit(self, guess, **kwargs):
        """Provide an initial guess for a nonlinear least-squared fit. All
        keyword arguments will be passed to scipy.optimize.minimize."""
        if kwargs.get('bounds', None) is None:
            bounds = [None]*len(guess)
        else:
            # If bounds are used, default to a method that accepts bounds.
            kwargs['method'] = kwargs.get('method', 'L-BFGS-B')
        result = optimize.minimize(self.residual, guess, **kwargs)
        return result.x

    def predict(self, exog, params):
        return self._predict(exog, params)

    def residual(self, *args):
        res = (self._predict(self.exog, *args) - self.endog)**2
        if self.weights is not None:
            res = np.multiply(res, self.weights)
        return res.sum()

class PowerFluid(Model):
    """Rotation angle of a wire in a power-law fluid 
       under a 90-degree step forcing. 
       Parameters: m, C=K/uB"""
    def _predict(self, theta, params):
        m, C, theta0, offset = map(np.real_if_close, params)
        return 1/(m-1)*C**m*(np.cos(theta + offset)**(1-m)* \
                             self.F(m, theta + offset) \
                             - np.cos(theta + offset)**(1-m)* \
                             self.F(m, theta0 + offset))

    @classmethod
    def F(cls, m, theta):
        "Convenience function"
        # _2F_1(1/2, (1-m)/2; (3-m)/2, cos^2(theta))
        return special.hyp2f1(0.5, (1-m)/2, (3-m)/2, np.cos(theta)**2)
        # return cls.recursive_series(m, theta)

    @classmethod
    def recursive_sequence(cls, m, theta, N=20):
        term = np.ones_like(theta)
        k = 0
        yield term
        while k < N:
            r = (1/2.+k)*((1-m)/2.+k)/((3-m)/2.+k)* np.cos(theta)**2/(k+1.)
            term = np.multiply(r, term)
            k += 1
            yield term

    @classmethod
    def recursive_series(cls, m, theta, N=20):
        return np.array(list(cls.recursive_sequence(m, theta, N))).sum(0)

class Viscous(Model):
    """Rotation angle of a wire in a viscous fluid under an arbitrary step forcing.
      Paramters: K, a, b"""
    def _predict(self, params):
        theta = self.exog
        K, a, b = map(np.real_if_close, params)
        return np.log(np.tan((theta - a)/b))/K
   
