import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from scipy import  ndimage

def threshold(im, sigma=3):
    """Threshold a grayscale image based on the mean and std brightness.

    Parameters
    ----------
    im: ndarray
    sigma: float, default 3.0
        minimum brightness in terms of standard deviations above the mean
    """
    mask = im > (im.mean() + sigma*im.std())
    return mask

def bigfish(mask, padding=0.03):
    """Identify the largest connected region and return the roi. 

    Parameters
    ----------
    mask: binary (thresholded) image
    padding: fractional padding of ROI (default 0.02)

    Returns
    -------
    padded_roi: a tuple of slice objects, for indexing the image
    """
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    big_label = sizes.argmax() # the label of the largest connection region
    roi = ndimage.find_objects(label_im==big_label)[0]
    padded_roi = pad_roi(roi, padding, mask.shape)
    return padded_roi

def pad_roi(roi, padding, img_shape):
    "Pad x and y slices, within the bounds of img_shape."
    s0, s1 = roi # slices in x and y
    p = int(np.max(img_shape)*padding)
    new_s0 = slice(np.clip(s0.start - p, 0, img_shape[0] - 1),
                   np.clip(s0.stop + p, 0, img_shape[0] - 1))
    new_s1 = slice(np.clip(s1.start - p, 0, img_shape[1] - 1),
                   np.clip(s1.stop + p, 0, img_shape[1] - 1))
    return new_s0, new_s1

def moment(img, i, j):
    """Utility function called by inertial_axes. See that function, below,
    for attribution and usage."""
    nrows, ncols = img.shape
    y, x = np.mgrid[:nrows, :ncols]
    return (img * x**i * y**j).sum()

def inertial_axes(img): 
    """Calculate the x-mean, y-mean, and cov matrix of an image.
    Parameters
    ----------
    img: ndarray
    
    Returns
    -------
    xbar, ybar, cov (the covariance matrix)

    Attribution
    -----------
    This function is based on a solution by Joe Kington, posted on Stack
    Overflow at http://stackoverflow.com/questions/5869891/
    how-to-calculate-the-axis-of-orientation/5873296#5873296
    """
    normalization = img.sum()
    m10 = moment(img, 1, 0)
    m01 = moment(img, 0, 1)
    x_bar = m10 / normalization 
    y_bar = m01 / normalization
    u11 = (moment(img, 1, 1) - x_bar * m01) / normalization
    u20 = (moment(img, 2, 0) - x_bar * m10) / normalization
    u02 = (moment(img, 0, 2) - y_bar * m01) / normalization
    cov = np.array([[u20, u11], [u11, u02]])
    return x_bar, y_bar, cov

def orientation(cov):
    """Compute the orientation angle of the dominant eigenvector of
    a covariance matrix.

    Parameters
    ----------
    cov: 2x2 array

    Returns
    -------
    angle in radians
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvec = eigvecs[eigvals.argmax()]
    return np.arctan2(eigvec[1], eigvec[0])

def analyze(frame, angle_only=True, plot=False):
    """Find a nanowire in a frame and return its orientation angle
    in degrees.

    Note
    ----
    This convenience function wraps several other functions with detailed
    docstrings. Refer to them for more information.

    Parameters
    ----------
    frame: image array
    angle_only: If True (default), return angle in degrees. If False,
       return x_bar, y_bar, cov -- the C.O.M. and the covariance matrix.
    plot: False by default. If True, plot principle axes over the ROI.
    """
    roi = bigfish(threshold(frame))
    blurred = ndimage.gaussian_filter(frame[roi].astype('float'), 3)
    masked = np.where(threshold(blurred, -0.5),
                     blurred, np.zeros_like(blurred))
    results = inertial_axes(masked)
    if plot:
        import mr.plots
        mr.plots.plot_principal_axes(frame[roi], *results)
    if angle_only:
        return np.rad2deg(orientation(results[2]))
    else:
        return results

def batch(frames, shift=True):
    count = frames.count
    data = Series(index=range(1, count + 1))
    for i, img in enumerate(frames):
        data[i + 1] = analyze(img)
    data = data.dropna() # Discard unused rows.
    if shift:
        data = data.where(data > 0, data + 180)
    return data
