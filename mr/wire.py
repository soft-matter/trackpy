import numpy as np
from scipy import  ndimage
from mr.kington import *

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

def bigfish(mask, padding=0.02):
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
    s0, s1 = roi # slices in x and y
    p = int(np.max(img_shape)*padding)
    new_s0 = slice(np.clip(s0.start - p, 0, img_shape[0] - 1),
                   np.clip(s0.stop + p, 0, img_shape[0] - 1))
    new_s1 = slice(np.clip(s1.start - p, 0, img_shape[1] - 1),
                   np.clip(s1.stop + p, 0, img_shape[1] - 1))
    return new_s0, new_s1

def filter(img, sigma=1):
    """Mask the original image using a gently thresholded image, in
    preparation for a fit."""
    return np.where(threshold(img, sigma), img, np.zeros_like(img))

def moment(data, i, j):
    """Utility function called by inertial_axes. See that function, below,
    for attribution and usage."""
    nrows, ncols = data.shape
    y, x = np.mgrid[:nrows, :ncols]
    return (data * x**i * y**j).sum()

def inertial_axes(data):
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
    how-to-calculate-the-axis-of-orientation/5873296#5873296"""

    normalization = data.sum()
    x_bar = moment(data, 1, 0) / normalization
    y_bar = moment(data, 0, 1) / normalization
    u11 = moment(data, 1, 1) / normalization - x_bar*y_bar
    u20 = moment(data, 2, 0) / normalization - x_bar**2
    u02 = moment(data, 0, 2) / normalization - y_bar**2
    cov = np.array([[u20, u11], [u11, u02]])
    return x_bar, y_bar, cov

def angle(cov):
    """Compute the orientation angle of the largest principal axis of
    a covariance matrix.

    Parameters
    ----------
    cov: 2x2 ndarray

    Returns
    -------
    angle in radians
    """

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvec = eigvecs[eigvals.argmax()] # dominant eigenvector
    angle = np.arctan2(eigvec[1], eigvec[0])
    return angle

