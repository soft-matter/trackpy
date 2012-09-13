import numpy as np
pi = np.pi

def fischer(D, a, contact_angle, bulk_visc=1e-3):
    """Use Fischer model to compute film viscosity from
    diffusivity D in um^2/s,
    sphere radius a in um,
    contact angle contact_angle,
    and bulk viscoity bulk_visc in SI (Pa s)."""
    d = a*(np.cos(contact_angle) - 1)
    c0 = 6*pi*np.sqrt(np.tanh(32*(d/a + 2)/(9*pi**2)))
    c1 = -4*np.log((2/pi)*np.arctan2((d + 2*a),(3*a)))
    kT = 4.1e-9 # Kg um^2 / s^2
    print c0, c1
    print "Review to verify units:"
    print "diffusivity = {} um^2 / s".format(D)
    print "radius = {} um".format(a)
    print "contact angle = {:.3f} radians = {:.0f} degrees".format(
           contact_angle, 180/pi*contact_angle)
    print "bulk viscosity = {} Pa s".format(bulk_visc)
    bulk_visc *= 1.e-6 # Kg / um s
    print kT/D, bulk_visc*a*c0
    film_visc = (kT/D - bulk_visc*a*c0)/c1 # Kg s = Pa m s 
    return film_visc
    
