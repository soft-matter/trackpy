import numpy as np
from scipy import special
from scipy import optimize
import pandas as pd

class Model(object):
    "Nonlinear model"
    def __init__(self, endog, exog, weights=None, sigma=None):
        # Remove missing values.
        self.exog = exog[endog.notnull()]
        self.endog = endog[exog.notnull()]
        self.endog = self.endog.dropna()
        self.exog = self.exog.dropna()
        assert len(self.exog) == len(self.endog)
        assert np.isfinite(self.endog).all()
        assert np.isfinite(self.exog).all()
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
        self.result = optimize.minimize(self.residual, guess, **kwargs)
        transformed_result = self.transform_vars(*self.result.x)
        return pd.Series(transformed_result, index=self.var_names)

    def line(self):
        return pd.Series(self.predict(self.exog, self.result.x))

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
       Parameters: m, C=K/uB
       Return: value, Jacobian"""
    

    @classmethod
    def _predict(cls, theta, params):
        m_, C_, theta0_, offset_ = params 
        m, C, theta0, offset = cls.transform_vars(m_, C_, theta0_, offset_)
        assert offset < 0, (
            "offset = {} > 0 will only lead to tears.").format(offset)
        assert offset > -np.pi/2, (
            "offset = {} < -pi/2 will only lead to tears.").format(offset)
        assert np.cos(theta0 + offset) > 0, (
            """"We require cos(theta0 + offset) > 0.
            theta0: {}
            offset: {}""".format(theta0, offset))
        assert C >= 0, (
            "C = {} < 0 is not physical.").format(C)
        assert m >= 0, (
            "m < 0 is not physical.").format(m)
        return cls.t(m, C, theta0, offset, theta)

    @classmethod
    def transform_vars(cls, m_, C_, theta0_, offset_):
        """Enforce bounds using analytical transformations."""
        m = np.abs(m_)
        C = np.abs(C_)
        offset = -(np.pi/2 - 0.0001)*0.5*(1 + np.tanh(offset_))
        theta0 = offset + (np.pi - 0.0001 - offset)*0.5*(1 + np.tanh(theta0_))
        cls.var_names = ['m', 'C', 'theta0', 'offset']
        return m, C, theta0, offset

    @classmethod
    def t_term(cls, m, C, offset, angle):
        """The function t consists of two similar terms like this.
        It is convenient to compute them individually."""
        term =  1/(m-1)*C**m*\
            np.cos(angle + offset)**(1-m)*cls.F(m, angle + offset)
        assert np.isfinite(term).all(), ("term not finite {}".format((m, C, offset, angle.min(), angle.max())))
        return term

    @classmethod
    def t(cls, m, C, theta0, offset, theta):
        "Time as function of m, C, and angles."""
        t_ = cls.t_term(m, C,  offset, theta) -\
            cls.t_term(m, C, offset, theta0)
        return t_

    @classmethod
    def jacobian(cls, m, C, theta0, offset, theta):
        """The analytical Jacobian of t, to help curve-fitting."""
        return (cls.dtdm(m, C, theta0, offset, theta),
            cls.dtdC(m, C, theta0, offset, theta),
            cls.dtdtheta0(m, C, theta0, offset, theta),
            cls.dtdoffset(m, C, theta0, offset, theta))

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
        # Or: return cls.recursive_series(m, theta)

    @classmethod
    def dtdm(cls, m, C, theta0, offset, theta):
        "First part of the Jacboian"
        # Compute the two terms of t seperately.
        t1 = cls.t_term(m, C, offset, theta)
        t2 = - cls.t_term(m, C, offset, theta)
        # Chain Rule:
        # The first two terms are of the form (something)*t.
        result = (-1./(m-1) + np.log(C))*(t1 + t2)
        # Third term
        result += -(np.log(np.cos(theta + offset))*t1 +\
            np.log(np.cos(theta0 + offset))*t2)
        # The last term involves dF/dm, which I compute using a series.
        result += 1./(m-1)*C**m*(np.cos(theta + offset)**(1-m)* \
            cls.dFdm(m, theta + offset) \
            - np.cos(theta0 + offset)**(1-m)* \
            cls.dFdm(m, theta0 + offset))
        return result

    @classmethod
    def dtdC(cls, m, C, theta0, offset, theta):
        "Second part of the Jacboian"
        # Proportional to t -- no need to compute t1, t2 separately.
        return m/C*cls.t(m, C, theta0, offset, theta)

    # Obviously, dt/d(theta0) and dt/d(offset) are similar.
    # The next two functions rely on dtdtheta(), whicch
    # does the similar work for each.

    @classmethod
    def dtdtheta0(cls, m, C, theta0, offset, theta):
        "Third part of the Jacobian"
        # Only the second term depends on the theta0.
        t2 = - cls.t_term(m, C, offset, theta0)
        return cls.dtdtheta(m, C, theta0 + offset, t2) 
        
    @classmethod
    def dtdoffset(cls, m, C, theta0, offset, theta):
        "Fourth part of the Jacobian"
        t1 = cls.t_term(m, C, offset, theta)
        t2 = - cls.t_term(m, C, offset, theta0)
        result = cls.dtdtheta(m, C, theta0 + offset, t1) 
        result += cls.dtdtheta(m, C, theta0 + offset, t2)
        return result

    @classmethod
    def dtdtheta(cls, m, C, angle, term):
        result = (1-m)/np.cos(angle)*term
        z = (np.cos(angle))**2 # convenient notation
        result += -C**m*np.cos(angle)**(1-m)/\
            (2*z)*(1/np.sqrt(1-z) - cls.F(m, angle))*\
            np.sin(2*angle)
        return result

    @classmethod
    def dFdm_sequence(cls, m, theta, N=10):
        from scipy.misc import factorial
        from scipy.special import psi # the digamma function
        for k in range(N+1):
            term = 1/(1 + 2*k/(1.-m))*factorial(2*k)/factorial(k)**2*\
                (np.cos(theta)/2.)**(2*k)*\
                ( (psi(1 + k + (1-m)/2.) - psi(1 + (1-m)/2.)) - \
                  (psi(k + (1-m)/2.) - psi((1-m)/2.)) )
            yield term

    @classmethod
    def dFdm(cls, m, theta, N=10):
        return np.array(list(cls.dFdm_sequence(m, theta, N))).sum(0)

    @classmethod
    def recursive_sequence(cls, m, theta, N=20):
        """Another way to compute F(m, theta).
        This sequence generator is summed by the function recursive_series."""
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
        """Another way to compute F(m, theta)."""
        return np.array(list(cls.recursive_sequence(m, theta, N))).sum(0)

    @classmethod
    def explicit_sequence(cls, m, theta, N=10):
        """Yet another way to compute F(m, theta). This one is not as clever.
        This sequence generator is summed by the function explicit_series."""
        from scipy.misc import factorial
        for k in range(N+1):
            term = 1/(1 + 2*k/(1. - m))*factorial(2*k)/factorial(k)**2*\
                (np.cos(theta)/2.)**(2*k)
            yield term

    @classmethod
    def explicit_series(cls, m, theta, N=10):
        """Yet another way to compute F(m, theta). This one is not as clever."""
        return np.array(list(cls.explicit_sequence(m, theta, N))).sum(0)

class Viscous(Model):
    """Rotation angle of a wire in a viscous fluid under an arbitrary step forcing.
      Paramters: K, a, b"""
    def _predict(self, params):
        theta = self.exog
        K, a, b = map(np.real_if_close, params)
        return np.log(np.tan((theta - a)/b))/K
   
