import re, os
import MySQLdb
from numpy import *
from scipy.ndimage.morphology import grey_dilation, grey_closing
from scipy.ndimage.filters import uniform_filter, generic_filter
from _Cfilters import nullify_secondary_maxima # custom-made
from scipy.ndimage.fourier import fourier_gaussian
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.interpolation import shift
from scipy.stats import scoreatpercentile
from itertools import groupby
from matplotlib.pyplot import *
from matplotlib import cm
from connect import sql_connect

# Make it possible to emulate Crocker/Grier/Weeks exactly.
# Whip up a few test cases to be sure that this is working.
# BIG IDEA: Get uncertainy as part of the result.

def bandpass(image, lshort, llong):
    """Convolve with a Gaussian to remove short-wavelength noise,
    and subtract out long-wavelength variations,
    retaining features of intermediate scale."""
    assert 2*lshort < llong, """The smoothing length scale must be more 
                              than twice the noise length scale."""
    smoothed_background = uniform_filter(image, 2*llong+1)
    no_noise = fft.ifft2(fourier_gaussian(fft.fft2(image), lshort))
    result = real(no_noise - smoothed_background)
    # Where result < 0 that pixel is definitely not a feature. Zero to simplify.
    return result.clip(min=0.)

def circular_mask(diameter, side_length):
    """A circle of 1's inscribed in a square of 0's,
    the 'footprint' of the features we seek."""
    r = int(diameter/2)
    L = int(side_length)
    mask = fromfunction(lambda x, y: sqrt((x-r)**2 + (y-r)**2), (L, L))
    mask[mask <= r] = True
    mask[mask > r] = False
    return mask

def local_maxima(image, diameter, separation, percentile=64, 
                 bigmask=None):
    """Local maxima whose brightness is above a given percentile.
    Passing masks to this function will improve performance of 
    repeated calls. (Otherwise, the masks are created for each call.)"""
    # Find the threshold brightness, representing the given
    # percentile among all NON-ZERO pixels in the image.
    flat = ravel(image)
    threshold = scoreatpercentile(flat[flat > 0], percentile)
    # The intersection of the image with its dilation gives local maxima.
    if bigmask is None:
        bigmask = circular_mask(diameter, separation)
    assert image.dtype == uint8, "Perform dilation on exact (uint8) data." 
    dilation = grey_dilation(image, footprint=bigmask)
    maxima = where((image == dilation) & (image > threshold))
    assert size(maxima) > 0, "Found zero maxima above the " + str(percentile) + \
                            "-percentile treshold at " + str(treshold) + "."
    # Flat peaks, for example, return multiple maxima.
    # Eliminate redundancies within the separation distance.
    berth = circular_mask(separation, separation)
    maxima_map = zeros_like(image)
    maxima_map[maxima] = image[maxima]
    peak_map = generic_filter(
        maxima_map, nullify_secondary_maxima(), 
        footprint=berth, mode='constant')
    # generic_filter calls a custom-built C function, for speed
    # Also, do not accept peaks near the edges.
    margin = int(floor(separation/2))
    peak_map[..., :margin] = 0
    peak_map[..., -margin:] = 0
    peak_map[:margin, ...] = 0
    peak_map[-margin:, ...] = 0
    peaks = where(peak_map != 0)
    assert size(peaks) > 0, "All maxima were in the margins."
    return [(x, y) for y, x in zip(*peaks)]

def estimate_mass(image, x, y, diameter, tightmask=None):
    "Find the total brightness in the neighborhood of a local maximum."
    # Define the square neighborhood of (x, y).
    r = int(floor(diameter/2))
    x0 = x - r; x1 = x + r + 1
    y0 = y - r; y1 = y + r + 1
    if tightmask is None:
        tightmask = circular_mask(diameter, diameter)
    # Take the circular neighborhood of (x, y).
    neighborhood = tightmask*image[y0:y1, x0:x1]
    return sum(neighborhood)

def refine_centroid(image, x, y, diameter, minmass=1, iterations=10,
                    tightmask=None, rgmask=None, thetamask=None,
                    sinmask=None, cosmask=None):
    """Characterize the neighborhood of a local 
    maximum to refine its center-of-brightness."""
    # Define the square neighborhood of (x, y).
    r = int(floor(diameter/2))
    x0 = x - r; x1 = x + r + 1
    y0 = y - r; y1 = y + r + 1
    if tightmask is None:
        tightmask = circular_mask(diameter, diameter)
    # Take the circular neighborhood of (x, y).
    neighborhood = tightmask*image[y0:y1, x0:x1]
    yc, xc = center_of_mass(neighborhood)  # neighborhood coordinates
    yc, xc = yc + y0, xc + x0  # image coordinates
    # Initially, the neighborhood is centered on the local max.
    # Shift it to the centroid, iteratively.
    ybounds = (0, image.shape[0] - 1 - 2*r)
    xbounds = (0, image.shape[1] - 1 - 2*r)
    assert iterations >= 1, "Set iterations=1 or more."
    for iteration in xrange(iterations):
        if (xc + r - x0 < 0.1 and yc + r - y0 < 0.1):
            break  # Accurate enough.
        # Start with whole-pixel shifts.
        if abs(xc - x0 - r) >= 0.6:
            x0 = clip(round(xc) - r, *xbounds)
            x1 = x0 + 2*r + 1
        if abs(yc -y0 -r) >= 0.6:
            y0 = clip(round(yc) - r, *ybounds)
            y1 = y0 + 2*r + 1
#       if abs(xc - x0 - r) < 0.6 and (yc -y0 -r) < 0.6:
            # Subpixel interpolation using a second-order spline.
#           shift(neighborhood,[yc, xc],mode='constant',cval=0., order=2)
        neighborhood = tightmask*image[y0:y1, x0:x1]    
        yc, xc = center_of_mass(neighborhood)  # neighborhood coordinates
        yc, xc = yc + y0, xc + x0  # image coordinates
    
    # Characterize the neighborhood of our final centroid.
    mass = sum(neighborhood)    
    if rgmask is None:
        rgmask = tightmask*fromfunction(lambda x, y: x**2 + y**2 + 1/6., (diameter, diameter))
    Rg2= sum(rgmask*image[y0:y1, x0:x1])/mass  # square of Rg 
    Rg = sqrt(Rg2)
    if thetamask is None or sinmask is None or cosmask is None:
        thetamask = tightmask*fromfunction(lambda y, x: arctan2(r-y,x-r), (diameter, diameter)) 
        sinmask = tightmask*sin(2*thetamask)
        cosmask = tightmask*cos(2*thetamask)
    ecc = sqrt((sum(neighborhood*cosmask))**2 + 
               (sum(neighborhood*sinmask))**2) / (mass - neighborhood[r, r] + 1e-6)
    return (xc, yc, mass, Rg2, ecc)

def merge_unit_squares(positions, separation, img_width):
    """Group all positions that are within the same square,
    sized by separation. Return one."""
    groups = []
    centroids = sorted(positions, 
                       key=lambda c: int(floor(c[0]/separation)) + 
                                     img_width*int(floor(c[1]/separation)))
    for key, group in groupby(positions, 
                              lambda c: int(floor(c[0]/separation)) + 
                                        img_width*int(floor(c[1]/separation))):
        groups.append(list(group))
    return [group[0] for group in groups]

def feature(image, diameter, separation=None, 
            percentile=64, minmass=1., pickN=None):
    "Locate circular Gaussian blobs of a given diameter."
    # Check parameters.
    assert diameter & 1, "Feature diameter must be an odd number. Round up."
    if not separation: separation = diameter + 1
    bigmask = circular_mask(diameter, separation)
    tightmask = circular_mask(diameter, diameter)
    rgmask = tightmask*fromfunction(lambda x, y: x**2 + y**2 + 1/6., (diameter, diameter))
    image = (255./image.max()*image.clip(min=0.)).astype(uint8)
    peaks = local_maxima(image, diameter, separation, percentile=percentile,
                         bigmask=bigmask)
    massive_peaks = [(x, y) for x, y in peaks if 
        estimate_mass(image, x, y, diameter, tightmask=tightmask) > minmass]
    centroids = [refine_centroid(image, x, y, diameter,
                 minmass=minmass, tightmask=tightmask, rgmask=rgmask) 
                 for x, y in massive_peaks]
    print len(peaks), 'local maxima', '  ', len(centroids), 'centroids'
    return centroids 

def locate(image_file, diameter, separation=None, 
           noise_size=1, smoothing_size=None, invert=True,
           percentile=64, minmass=1., pickN=None):
    "Wraps feature with image reader, black-white inversion, and bandpass."
    if not smoothing_size: smoothing_size = diameter
    image = imread(image_file)
    if invert:
        image = 1 - image
    image = bandpass(image, noise_size, smoothing_size)
    return feature(image, diameter, separation=separation,
                   percentile=percentile, minmass=minmass,
                   pickN=pickN)

def batch(trial, stack, image_file_list, diameter, separation=None,
          noise_size=1, smoothing_size=None, invert=True,
          percentile=64, minmass=1., pickN=None, override=False):
    """Analyze a list of image files, and insert the centroids into
    the database."""
    conn = sql_connect()
    if sql_duplicate_check(trial, stack, conn):
        if override:
            print 'Overriding'
        else:
            print 'There are entries for this trial and stack already.'
            conn.close()
            return False
    for frame, filepath in enumerate(image_file_list):
        frame += 1 # Start at 1, not 0.
        centroids = locate(filepath, diameter, separation, noise_size,
               smoothing_size, invert, percentile, minmass, pickN)
        sql_insert(trial, stack, frame, centroids, conn, override)
    conn.close()

def sql_duplicate_check(trial, stack, conn):
    "Return false if the database has no entries for this trial and stack."
    c = conn.cursor()
    c.execute("SELECT COUNT(1) FROM Features WHERE trial=%s AND stack=%s",
              (trial, stack))
    count, = c.fetchone()
    return count != 0.0

def sql_insert(trial, stack, frame, centroids, conn, override=False):
    "Insert centroid information into the MySQL database."
    try:
        c = conn.cursor()
        # Load the data in a small temporary table.
        c.execute("CREATE TEMPORARY TABLE NewFeatures"
                  "(x float, y float, mass float, size float, ecc float)")
        c.executemany("INSERT INTO NewFeatures (x, y, mass, size, ecc) "
                      "VALUES (%s, %s, %s, %s, %s)", centroids)
        # In one step, tag all the rows with identifiers (trial, stack, frame).
        # Copy the temporary table into the big table of features.
        c.execute("INSERT INTO Features "
                  "(trial, stack, frame, x, y, mass, size, ecc) "
                  "SELECT %s, %s, %s, x, y, mass, size, ecc FROM NewFeatures" 
                  % (trial, stack, frame))
        c.execute("DROP TEMPORARY TABLE NewFeatures")
        c.close()
    except:
        print sys.exc_info()
        return False
    return True

def annotate(image, positions, output_file=None, 
             circle_size=170, delay_show=False):
    "Draw white circles on the image, like Eric Weeks' fover2d."
    # The parameter image can be an image object or a filename.
    if type(image) is str:
	image = 1-imread(image)
    x, y = array(positions)[:,0:2].T
    clf()
    imshow(image, origin='upper', shape=image.shape, cmap=cm.gray)
    xlim(0, image.shape[1])
    ylim(0, image.shape[0]) 
    scatter(x, y, s=circle_size, facecolors='none', edgecolors='w')
    if output_file:
        savefig(output_file)
    elif not delay_show:
        show() 

def sample(image_file_list, diameter, separation=None,
           noise_size=1, smoothing_size=None, invert=True,
           percentile=64, minmass=1., pickN=None):
    """Try parameters on a small sampling of images (out of potenitally huge
    list). Show annotated images."""
    samples = [image_file_list[0], 
               image_file_list[len(image_file_list)/2], 
               image_file_list[-1]] # first, middle, last
    for i, image_file in enumerate(samples):
        print "Sample " + str(1+i) + " of " + str(len(samples)) + "..."
        f = locate(image_file, diameter, separation,
                   noise_size, smoothing_size, invert,
                   percentile, minmass, pickN)
        annotate(image_file, f, delay_show=True)
        show()

def list_images(directory):
    "List the path to all image files in a directory."
    files = os.listdir(directory)
    images = [os.path.join(directory, f) for f in files if \
        os.path.isfile(os.path.join(directory, f)) and re.match('.*\.png', f)]
    return sorted(images)

def batch_annotate(trial, stack, directory, new_directory):
    """Save annotated copies of analyzed frames, referring to the database
    to retrieve feature positions."""
    imlist = list_images(directory)
    conn = connect()
    for frame, filepath in enumerate(imlist):
        frame += 1 # Start at 1, not 0.
        c = conn.cursor()
        c.execute("SELECT x, y FROM Features WHERE trial=%s AND stack=%s AND "
                  "frame=%s", (trial, stack, frame))
        positions = array(c.fetchall())
        filename = os.path.basename(filepath)
        output_file = os.path.join(new_directory, filename)
        annotate(filepath, positions, output_file=output_file)
    conn.close()
