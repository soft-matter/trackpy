import numpy as np
from scipy import special
import statsmodels
from statsmodels.miscmodels.nonlinls import NonlinearLS
import pandas as pd

def dropna(func):
    """Intercept data on its way to a fitting model, and remove the Nans."""
    def wrapper(*args, **kwargs):
        endog = kwargs.get('endog', args[1])
        exog = kwargs.get('exog', args[2])
        together = pd.concat([pd.DataFrame(endog), pd.DataFrame(exog)], axis=1)
        cleaned = together.dropna(axis=0)
        new_endog = pd.Series(cleaned.ix[:, 0])
        new_exog = pd.Series(cleaned.ix[:, 1])
        kwargs['endog'] = new_endog
        kwargs['exog'] = new_exog
        return func(args[0], **kwargs)
    return wrapper

class PowerFluid(NonlinearLS):
    """Rotation angle of a wire in a power-law fluid 
       under a 90-degree step forcing. 
       Parameters: m, C=K/uB"""
    def _predict(self, params):
        theta = self.exog
        return self.func(params, theta)

    @staticmethod
    def F(m, x):
        "Convenience function"
        # _2F_1(1/2, (1-m)/2; (3-m)/2, cos^2(theta))
        return special.hyp2f1(0.5, (1-m)/2, (3-m)/2, np.cos(x)**2)

    @classmethod
    def func(self, params, theta):
        m, C = map(np.real_if_close, params)
        return 1/(m-1)*C**m*(np.cos(theta)**(1-m)*self.F(m, theta) \
                             - np.cos(theta)**(1-m)*self.F(m, 0))

    __init__ = dropna(NonlinearLS.__init__)

class FixedPowerFluid(NonlinearLS):
    """Rotation angle of a wire in a power-law fluid under a 90-degree step forcing. 
       Set m like so:
       >>> model = FixedPowerFluid
       >>> model.m = 2

       Parameters: m, C=K/uB"""
    def _predict(self, params):
        theta = self.exog
        return self.func(params, theta)

    @staticmethod
    def F(m, x):
        "Convenience function"
        # _2F_1(1/2, (1-m)/2; (3-m)/2, cos^2(theta))
        return special.hyp2f1(0.5, (1-m)/2, (3-m)/2, np.cos(x)**2)

    @classmethod
    def func(self, params, theta):
        m = self.m
        C, = map(np.real_if_close, params)
        return 1/(m-1)*C**m*(np.cos(theta)**(1-m)*self.F(m, theta) \
                             - np.cos(theta)**(1-m)*self.F(m, 0))

    __init__ = dropna(NonlinearLS.__init__)

    @property
    def m(self):
        return self._

    @m.setter
    def m(self, value):
        self._m = value

    __init__ = dropna(NonlinearLS.__init__)
        
class Viscous(NonlinearLS):
    """Rotation angle of a wire in a viscous fluid under an arbitrary step forcing.
      Paramters: K, a, b"""
    def _predict(self, params):
        theta = self.exog
        K, a, b = params
        return np.log(np.tan((theta - a)/b))/K
    
   # __init__ = dropna(NonlinearLS.__init__)
