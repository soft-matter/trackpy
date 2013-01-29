import numpy as np
import pandas as pd
from scipy import special
import lmfit
from model_utils import params_as_dict

def linearized_HB(angle, params):
    m = params['m']
    C = params['C']
    A = params['A']
    t = C**m*np.cos(angle)**(-m)*\
        (A*special.hyp2f1(1/2., -m/2., (2-m)/2., np.cos(angle)**2) + \
            np.cos(angle)/(m-1)*_F(angle, m)) -\
        C**m*(np.sqrt(np.pi)*A*special.gamma(1-m/2.)/special.gamma((1-m)/2.) +
              np.sqrt(np.pi)*special.gamma((3-m)/2.)/((m-1)*special.gamma(1-m/2.)))
    return t

def _F(angle, m):
    "Convenience function"
    # _2F_1(1/2, (1-m)/2; (3-m)/2, cos^2(theta))
    result = special.hyp2f1(0.5, (1-m)/2, (3-m)/2, np.cos(angle)**2)
    assert np.isfinite(result).all(), (
        """Hypergeometric function returned a result that is not finite.
        m={}
        result={}""".format(m, result))
    return result
