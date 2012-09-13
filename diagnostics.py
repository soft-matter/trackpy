import matplotlib.pyplot as plt
import numpy as np

def annotate(image, positions, circle_size=170):
    "Draw white circles on the image, like Eric Weeks' fover2d."
    # The parameter image can be an image object or a filename.
    if type(image) is str:
	image = 1-plt.imread(image)
    x, y = np.array(positions)[:,0:2].T
    plt.imshow(image, origin='upper', shape=image.shape, cmap=plt.cm.gray)
    plt.xlim(0, image.shape[1])
    plt.ylim(0, image.shape[0]) 
    plt.scatter(x, y, s=circle_size, facecolors='none', edgecolors='w')
    plt.show()

def subpx_hist(positions):
    """Historgram the decimal parts of x and y. They should be flat.
    If not, you probably do not have good sub-pixel accuracy."""
    x, y = np.array(positions)[:, 0:2].T
    fracx, fracy = np.modf(x)[0], np.modf(y)[0]
    plt.hist(fracx, bins=np.arange(0, 1.1, 0.1), color='#667788', label='x mod 1.0')
    plt.hist(fracy, bins=np.arange(0, 1.1, 0.1), color='#994433', alpha=0.5, label='y mod 1.0')
    plt.legend()
    plt.show()
