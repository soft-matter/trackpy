import numpy as np
from scipy import special
from scipy import optimize
import pandas as pd
from pandas import DataFrame, Series
pi = np.pi

def box(x, a, b, cushion=1.e-6):
    """Enforce bounds using an analytical transformation."""
    assert cushion > 0, "cushion must be an absolute value"
    assert a < b, "Required: a < b"
    unit_box = (1 + np.tanh(x))/2. # from 0 to 1
    return a + cushion + (b - a - 2*cushion)*unit_box

def unbox(x, a, b, cushion=1.e-6):
    """Undo box."""
    assert cushion > 0, "cushion must be an absolute value"
    assert x < b, "Required: a < x < b"
    assert x > a, "Required: a < x < b"
    return np.arctanh(2*(x - a - cushion)/(b - a - 2*cushion) - 1)

def transform(angle, m_, C_, theta0_, offset_):
    "Bound m and C below. Bound theta0 and offset above and below."
    m = np.abs(m_)
    C = np.abs(C_)
    offset = box(offset_, -pi/2 - angle.min(), pi/2 - angle.max())
    theta0 = box(theta0_, -pi/2 - offset, pi/2 - offset) 
    return m, C, theta0, offset

def untransform(angle, m_, C_, theta0_, offset_):
    m = m_
    C = C_
    offset = unbox(offset_, -pi/2 - angle.min(), pi/2 - angle.max())
    theta0 = unbox(theta0_, -pi/2 - offset, pi/2 - offset) 
    return m, C, theta0, offset

def power_fluid(angle, m, C, theta0, offset):
    m, C, theta0, offset = transform(angle, m, C, theta0, offset)
    validate(angle, m, C, theta0, offset)
    # print 'type:', type(angle)
    # print 'extremes:', angle.min(), angle.max()
    # print 'transformed and validated params:', m, C, theta0, offset
    term1 = 1/(m-1)*C**m*\
        np.cos(angle + offset)**(1-m)*F(angle + offset, m)
    term2 = 1/(m-1)*C**m*\
        np.cos(theta0 + offset)**(1-m)*F(theta0 + offset, m)
    return term1 + term2

def validate(angle, m, C, theta0, offset):
    assert C >= 0, (
        "C = {} < 0 is not physical.").format(C)
    assert m >= 0, (
        "m < 0 is not physical.").format(m)
    assert m != 1, (
        """m == 1 means that the flow index n is also 1.
           This model diverges, but a purely viscous one should work.""")
    assert np.all(np.cos(angle + offset) >= 0), "cos(angle + offset) < 0"
    assert np.all(np.cos(angle + offset) >= 0), "cos(angle + offset) < 0"
    assert np.all(np.cos(theta0 + offset) >= 0), "cos(theta0 + offset) < 0"

def F(angle, m):
    "Convenience function"
    # _2F_1(1/2, (1-m)/2; (3-m)/2, cos^2(theta))
    result = special.hyp2f1(0.5, (1-m)/2, (3-m)/2, np.cos(angle)**2)
    assert np.isfinite(result).all(), (
        """Hypergeometric function returned a result that is not finite.
        m={}
        result={}""".format(m, result))
    return result

def transform_fits(angle, fits):
    """Apply same transformation to final result, which will be given
    in weird varibles."""
    transformed_fits = fits.T.apply(
        lambda x: Series(transform(angle[x.name], *x), index=x.index))
    return transformed_fits
