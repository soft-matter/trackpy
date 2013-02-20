from PIL import Image
import numpy as np
from scipy import stats
import lmfit

def crop_roi(im, roi):
    im = Image.open(filename)
    im = im.crop(roi_coords).load() # left, upper, right, lower
    return im

def _residual(params, x, y):
    A = params['A'].value
    base = params['base'].value
    sigma = params['sigma'].value
    x0 = params['x0'].value
    
    f = base + A*np.exp(-(x-x0)**2/sigma)
    return y - f

def _gaussian_center(params, x, data):
    lmfit.minimize(_residual, params, args=(x, data))
    return params['x0'].value, params['A'].value

def get_centers(im):
    L = im.shape[1]
    params = lmfit.Parameters()
    params.add('A', 100, min=0, max=255)
    params.add('sigma', 3, min=0, max=L/2)
    params.add('x0', L/2, min=0, max=L)
    params.add('base', 1, min=0, max=254)

    gaussian_center = lambda a: _gaussian_center(params, np.arange(L), a)
    fit = DataFrame(np.apply_along_axis(gaussian_center, 1, im), columns=['x0', 'A'])
    return fit

def get_angle(centers, threshold=20):
    real_centers = centers[centers['A'] > threshold]['x0'] # Filter junk.
    slope, intercept, r, p, stderr = \
            stats.linregress(real_centers.index, real_centers.values)
    return np.degrees(np.arctan(slope))
