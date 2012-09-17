import logging
import numpy as np
pi = np.pi

logger = logging.getLogger(__name__)

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
    
def gse(t, r2, a, T=298, width=0.7):
    """Compute G from Generalized Stokes-Einstein."""
    kT = 4.1e-9*T/298. # Kg um^2 / s^2

def _log_deriv(x, f, width):
    smooth_f = _parabolic_spline(np.log(x), np.log(f))
    df[i] = 2*a*logx[i] + b
    ddf[i] = 2*a

def _parabolic_spline(x, f, width):
    weights = np.exp(-(x -x[i])**2/(2*width**2))
    a, b, c = np.polyfit(x, f, deg=2, w=weights)
    return a*x[i]**2 + b*x[i] + c
