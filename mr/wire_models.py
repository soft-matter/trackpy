import numpy as np
from scipy import special
from scipy import optimize
import pandas as pd

class Model(object):
    "Nonlinear model"
    def __init__(self, endog, exog, weights=None, sigma=None):
        # Remove missing values.
        self.exog = exog[endog.notnull()]
        self.endog = endog.dropna()
        assert len(self.exog) == len(self.endog)
        assert self.exog.notnull().all()
        assert self.endog.notnull().all()
        if weights is not None:
            self.weights = weights
        elif sigma:
            self.weights = 1./sigma
        else:
            self.weights = None

    def fit(self, guess, **kwargs):
        """Provide an initial guess for a nonlinear least-squared fit. All
        keyword arguments will be passed to either scipy.optimize.leastsq
        or scipy.optimize.minimize, depending on the choice of 'method'."""
        if kwargs.get('method') == 'LM':
            # Use scipy.optimize.leastsq, which implements the
            # Levenburg-Marquardt algorithm.
            del kwargs['method'] # Remaining kwargs are passed to leastsq.
            result = optimize.leastsq(self.error, guess, **kwargs)
            return result
        if kwargs.get('bounds'):
            # If bounds are used, default to a method that accepts bounds.
            kwargs['method'] = kwargs.get('method', 'L-BFGS-B')
        result = optimize.minimize(self.residual, guess, **kwargs)
        return result.x

    @classmethod
    def predict(cls, exog, params):
        return cls._predict(exog, params)

    def error(self, *args):
        self.check_args(*args)
        prediction = self._predict(self.exog, *args)
        self.check_prediction(prediction, *args)
        err = (prediction - self.endog)**2
        if self.weights is not None:
            err = np.multiply(err, np.sqrt(self.weights))
        self.check_err(err, prediction, *args)
        return err

    def residual(self, *args):
        err = self.error(*args)
        res = (err**2).sum()
        return res

    def check_args(self, *args):
        assert np.isfinite(np.array(*args)).all(), (
            "Woah there. I have started making guesses that "
            "are not finite.")

    def check_prediction(self, prediction, *args):
        assert np.isfinite(prediction).all(), (
            """Prediction is not finite!
            Args: {}
            Prediction: {}""".format(args, prediction))

    def check_err(self, err, prediction, *args):
        assert np.isfinite(err).all(), (
            """Error array is not finite!
            Args: {}
            Error: {}""".format(args, prediction, err))

class PowerFluid(Model):
    """Rotation angle of a wire in a power-law fluid 
       under a step forcing. 
       Parameters: m, C=K/uB"""
    @classmethod
    def _predict(cls, theta, params):
        m, C, theta0, offset = params 
        offset = 0
        assert offset <=0, (
            "offset = {} > 0 will only lead to tears.").format(offset)
        assert offset > -np.pi/2, (
            "offset = {} < -pi/2 will only lead to tears.").format(offset)
        assert np.cos(theta0 + offset) > 0, "We require cos(theta0 + offset) > 0."
        assert C >= 0, (
            "C = {} < 0 is not physical.").format(C)
        assert m >= 0, (
            "m < 0 is not physical.").format(m)
        return 1./(m-1)*C**m*(np.cos(theta + offset)**(1-m)* \
                             cls.F(m, theta + offset) \
                             - np.cos(theta0 + offset)**(1-m)* \
                             cls.F(m, theta0 + offset))

    @classmethod
    def F(cls, m, theta):
        "Convenience function"
        # _2F_1(1/2, (1-m)/2; (3-m)/2, cos^2(theta))
        values = special.hyp2f1(0.5, (1-m)/2, (3-m)/2, np.cos(theta)**2)
        assert np.isfinite(values).all(), (
            """Hypergeometric function returned values that are not finite.
            m={}
            values={}""".format(m, values))
        return values 
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
   
