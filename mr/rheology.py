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

import logging
import numpy as np
from scipy import special
pi = np.pi

logger = logging.getLogger(__name__)

def fit_powerlaw(data):
    data = DataFrame(data)
    fits = []
    for col in data:
        slope, intercept, r, p, stderr = \
            stats.linregress(data.index.values, data[col].values)
        fits.append(Series([slope, np.exp(intercept)], index=['A', 'n']))
    return pd.concat(fits, axis=1, keys=data.columns)

def fischer(D, a, contact_angle, bulk_visc=1e-3, T=298):
    """Use Fischer model to compute film viscosity from:
    diffusivity D in um^2/s,
    sphere radius a in um,
    contact angle contact_angle,
    and bulk viscoity bulk_visc in SI (Pa s).
    Result is returned in SI, Pa m s = Kg s."""
    d = a*(np.cos(contact_angle*pi/180) - 1)
    c0 = 6*pi*np.sqrt(np.tanh(32*(d/a + 2)/(9*pi**2)))
    c1 = -4*np.log((2/pi)*np.arctan2((d + 2*a), (3*a)))
    kT = 4.1e-9*T/298. # Kg um^2 / s^2
    logger.info("c0=%.5f, c1=%.5f", c0, c1)
    logger.info("Review the following to verify units...\n"
                 "D = %.3f um^2 / s\n"
                 "radius = %.3f um\n"
                 "contact angle = %.0f degrees = %.3f radians\n"
                 "bulk viscosity = %f Pa s",
                 D, a, contact_angle, contact_angle*pi/180, bulk_visc)
    bulk_visc *= 1.e-6 # Kg / um s
    film_visc = (kT/D - bulk_visc*a*c0)/c1 # Kg s = Pa m s 
    return film_visc
    
def gse(t, r2, a, T=298, clip=0.03, width=0.7):
    """Compute G*(w) from r^2(t) using Generalized Stokes-Einstein.
    See T.G. Mason et. al. doi:10.1103/PhysRevLett.79.3282."""
    kT = 4.1e-9*T/298. # Kg um^2 / s^2
    DIM = 2 # dimensions of 
    s = w = 1./t
    f, df, ddf = log_derivatives(t, r2)
    if np.abs(ddf).max() > 0.15:
        logger.warning("Second logrithmic derivative of the MSD "
                       "reaches %.2f. Data is not very "
                       "power-law like. Results may be poor.", 
                       np.abs(ddf).max())
    G_s = (DIM/3.)*kT/(pi*a) / (f*special.gamma(1 + df)*(1 + ddf/2.))
    g, dg, ddg = log_derivatives(s, G_s)
    if np.abs(ddg).max() > 0.15:
        logger.warning("Second-order logrithmic derivative of G(s) "
                       "reaches %.2f. Data is not very "
                       "power-law like. Results may be poor.", 
                       np.abs(ddg).max())
    Gp = g/(1. + ddg) * (np.cos(pi/2*dg) - (pi/2-1)*dg*ddg)
    Gpp = g/(1. + ddg) * (np.sin(pi/2*dg) - (pi/2-1)*(1 - dg)*ddg)
    # If G'(w) or G''(w) are less than 3% of G(w), the signal is probably
    # buried by the noise.
    if np.any(Gp < clip*G_s):
        Gp = np.ma.array(Gp, mask=(Gp < clip*G_s), fill_value=0.)
        logger.info("Some values of G' << G. A masked "
                    "array will be returned.")
    if np.any(Gpp < clip*G_s):
        Gpp = np.ma.array(Gpp, mask=(Gpp < clip*G_s), fill_value=0.)
        logger.info("Some values of G'' << G. A masked "
                    "array will be returned.")
    return w, G_s, Gp, Gpp

def log_derivatives(x, f, width=0.7):
    """In the neighborhood of each x, approximate f(x) as a parabola 
    and logrithmically differentiate the parabola.
    Return this approximated f(x), d(log f)/d(log x), 
    and d^2(log f)/d(log x)^2."""
    assert len(x) == len(f), "x and f must be the same length."
    logx, logf = np.log(x), np.log(f)
    smooth_f = np.zeros_like(f)
    df = np.zeros_like(f)
    ddf = np.zeros_like(f)
    for i, x0 in enumerate(logx):
        c, b, a = _parabolic_spline(logx, logf, x0, width)
        smooth_f[i] = np.exp(a*x0**2 + b*x0 + c)
        df[i] = 2*a*x0 + b
        ddf[i] = 2*a
    return smooth_f, df, ddf
        
def _parabolic_spline(x, f, x0, width):
    """Fit a parabola to f(x) in the neighborhood of x0.
    Return the coefficients c, b, a as in f(x) = ax^2 + bx + c."""
    weights = np.exp(-(x - x0)**2/(2*width**2)) # Gaussian
    c, b, a = np.polynomial.polynomial.polyfit(x, f, deg=2, w=weights)
    # Warning: There is a different function called np.polyfit that does not
    # accept weights. This long namespace calls the weight-capable polyfit.
    return c, b, a

def toy_data(n):
    "Return powerlaw 'MSD data' with power n."
    return np.arange(1, 100), np.arange(1, 100)**n+0.001*np.random.rand()
