import numpy as np
from scipy import  ndimage

def threshold(im, sigma=3):
    """Threshold a grayscale image based on the mean and std brightness.

    Parameters
    ----------
    im: ndarray
    sigma: float, default 3.0
        minimum brightness in terms of standard deviations above the mean
    """
    mask = im > (im.mean() + 3*im.std())
    return mask

def bigfish(mask):
    """Identify the largest connected region and return the roi. 

    Parameters
    ----------
    mask: binary (thresholded) image
    """
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    big_label = sizes.argmax() # the label of the largest connection region
    roi = ndimage.find_objects(label_im==big_label)[0]
    return roi
