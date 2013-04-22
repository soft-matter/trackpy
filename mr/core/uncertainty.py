import numpy as np
from scipy.ndimage import morphology
from mr.core.preprocessing import bandpass, circular_mask

def roi(image, diameter):
    """Return a mask selecting the neighborhoods of bright regions.
    See Biophysical journal 88(1) 623-638 Figure C.

    Parameters
    ----------
    image : ndarray
    diameter : feature size used for centroid identification

    Returns
    -------
    boolean ndarray, True around bright regions
    """
    signal_mask = bandpass(image, 1, 2*diameter + 1)
    structure = circular_mask(diameter)
    signal_mask = morphology.binary_dilation(signal_mask, structure=structure)
    return signal_mask

def measure_noise(image, diameter):
    "Compute the standard deviation of the dark pixels outside the signal."
    signal_mask = roi(image, diameter)
    return image[~signal_mask].std()

def static_error(features, noise, diameter, noise_size=1):
    """Compute the uncertainty in particle position ("the static error").

    Parameters
    ----------
    features : DataFrame of features (or trajectories) including signal and size
    noise : Series of noise measurements, indexed by frame
    diameter : feature diameter used to locate centroids
    noise_size : half-width of Gaussian blurring used in image preparation

    Returns
    -------
    Series of static error estimates, indexed like the trajectories

    Note
    ----
    This is based on the process described by Thierry Savin and Patrick S. Doyle in their
    paper "Static and Dynamic Errors in Particle Tracking Microrheology,"
    Biophysical Journal 88(1) 623-638. Any mistakes are due to me, Daniel Allan.
    """
    # If this is just one frame, noise is a scalar.
    if np.isscalar(noise):
        N_S = noise/features['signal']
    # Otherwise, join by frame number.
    else:
        noise.name = 'noise'
        N_S = features.join(noise, on='frame')['noise']/features['signal']
    ep = N_S*noise_size/(2*np.pi**0.5)*(diameter/features['size'])**2 # Savin & Doyle, Eq. 55
    ep.name = 'ep' # so it can be joined
    return ep

