from mr.core import feature
import numpy as np
from scipy.ndimage import morphology

def roi(features, image, diameter, separation=None, percentile=64):
    separation = 1 + diameter if separation is None else separation
    feature_map = np.zeros_like(image, dtype='bool')
    feature_map[(features.y.astype('int'), features.x.astype('int'))] = True 
    dilation = morphology.binary_dilation(image)
    return dilation
