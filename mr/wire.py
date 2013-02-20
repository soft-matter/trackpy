from PIL import Image
import lmfit

def crop_roi(im, roi):
    im = Image.open(filename)
    im = im.crop(roi_coords).load() # left, upper, right, lower
    return im

def roughly_orient(im, angle):
    return im.rotate(angle)

def fit(im):
    A = lmfit.Parameter('A', 0.5, min=0, max=1)
    sigma = lmfit.Parameter('sigma', 3, min=0)
    x0 = lmfit.Parameter('x0', im.size[0])
    base = lmfit.Parameter('base', 0, min=0, max=1)

    def residual(params, x, y):
        f = params['base'] + params['A']*np.exp(x**2/params['sigma']))
        return y - f

    x = np.arange(im.size[0])
    lmfit.minimize(residual, params, args=(x, data))
    params['x0'].value
