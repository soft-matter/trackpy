"""These functions generate handy plots."""

from functools import wraps
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
import motion

logger = logging.getLogger(__name__)

def make_axes(func):
    """
    A decorator for plotting functions.
    NORMALLY: Direct the plotting function to the current axes, gca().
              When it's done, make the legend and show that plot.
              (Instant gratificaiton!)
    BUT:      If the uses passes axes to plotting function, write on those axes
              and return them. The user has the option to draw a more complex
              plot in multiple steps.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get('ax') is None:
            kwargs['ax'] = plt.gca()
            # Delete legend keyword so remaining ones can be passed to plot().
            try:
                legend = kwargs['legend']
            except KeyError:
                legend = None
            else:
                del kwargs['legend']
                print 'hi'
            result = func(*args, **kwargs)
            if not (kwargs['ax'].get_legend_handles_labels() == ([], []) or \
                    legend is False):
                plt.legend(loc='best')
            plt.show()
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

def make_fig(func):
    """See make_axes."""
    wraps(func)
    def wrapper(*args, **kwargs):
        if 'fig' not in kwargs:
            kwargs['fig'] = plt.gcf()
            func(*args, **kwargs)
            plt.show()
        else:
            return func(*args, **kwargs)
    return wrapper

@make_axes
def plot_traj(traj, colorby='particle', mpp=1, label=False, superimpose=None,
       cmap=mpl.cm.winter, ax=None):
    """Plot traces of trajectories for each particle.
    Optionally superimpose it on a frame from the video.

    Parameters
    ----------
    traj : DataFrame including columns x and y
    colorby: {'particle', 'frame'}
    mpp : microns per pixel
    label : Set to True to write particle ID numbers next to trajectories.
    superimpose : background image, default None
    cmap : This is only used in colorby='frame' mode.
        Default = mpl.cm.winter
    ax : matplotlib axes object, defaults to current axes

    Returns
    -------
    None
    """
    if (superimpose is not None) and (mpp != 1):
        raise NotImplementedError("When superimposing over an image, you " +
                                  "must plot in units of pixels. Leave " +
                                  "microns per pixel mpp=1.")
    # Axes labels
    if mpp == 1:
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')
    else:
        ax.set_xlabel(r'x [$\mu$m]')
        ax.set_ylabel(r'y [$\mu$m]')
    # Background image
    if superimpose is not None:
        ax.imshow(superimpose, cmap=plt.cm.gray)
        ax.set_xlim(0, superimpose.shape[1])
        ax.set_ylim(0, superimpose.shape[0])
    # Trajectories
    if colorby == 'particle':
        # Unstack particles into columns.
        unstacked = traj.set_index(['frame', 'particle']).unstack()
        ax.plot(mpp*unstacked['x'], mpp*unstacked['y'], linewidth=1)
    if colorby == 'frame':
        # Read http://www.scipy.org/Cookbook/Matplotlib/MulticoloredLine
        from matplotlib.collections import LineCollection
        x = traj.set_index(['frame', 'particle'])['x'].unstack()
        y = traj.set_index(['frame', 'particle'])['y'].unstack()
        color_numbers = traj['frame'].values/float(traj['frame'].max())
        logger.info("Drawing multicolor lines takes awhile. "
                    "Come back in a minute.")
        for particle in x:
            points = np.array(
                [x[particle].values, y[particle].values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap)
            lc.set_array(color_numbers)
            ax.add_collection(lc)
            ax.set_xlim(x.apply(np.min).min(), x.apply(np.max).max())
            ax.set_ylim(y.apply(np.min).min(), y.apply(np.max).max())
    if label:
        unstacked = traj.set_index(['frame', 'particle'])[['x', 'y']].unstack()
        first_frame = int(traj['frame'].min())
        coords = unstacked.fillna(method='backfill').stack().loc[first_frame]
        for particle_id, coord in coords.iterrows():
            plt.text(coord['x'], coord['y'], "%d" % particle_id,
                     horizontalalignment='center',
                     verticalalignment='center')
    ax.invert_yaxis()
    return ax

ptraj = plot_traj # convenience alias

@make_axes
def annotate(centroids, image, circle_size=170, color='g',
             invert=False, ax=None):
    """Mark identified features with white circles.

    Parameters
    ----------
    centroids : DataFrame including columns x and y
    image : image array (or string path to image file)
    circle_size : size of circle annotations in matplotlib's annoying
        arbitrary units, default 170
    color : string
        default 'g'
    invert : If you give a filepath as the image, specify whether to invert
        black and white. Default True.
    ax : matplotlib axes object, defaults to current axes

    Returns
    ------
    axes
    """
    # The parameter image can be an image object or a filename.
    if isinstance(image, basestring):
        image = plt.itpead(image)
    if invert:
        ax.imshow(1-image, origin='upper', shape=image.shape, cmap=plt.cm.gray)
    else:
        ax.imshow(image, origin='upper', shape=image.shape, cmap=plt.cm.gray)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.scatter(centroids['x'], centroids['y'],
               s=circle_size, facecolors='none', edgecolors=color)
    return ax

@make_axes
def mass_ecc(f, ax=None):
    """Plot each particle's mass versus eccentricity."""
    ax.plot(f['mass'], f['ecc'], 'ko', alpha=0.3)
    ax.set_xlabel('mass')
    ax.set_ylabel('eccentricity (0=circular)')
    return ax

@make_axes
def mass_size(f, ax=None):
    """Plot each particle's mass versus size."""
    ax.plot(f['mass'], f['size'], 'ko', alpha=0.1)
    ax.set_xlabel('mass')
    ax.set_ylabel('size')
    return ax

@make_axes
def subpx_bias(f, ax=None):
    """Histogram the fractional part of the x and y position.

    Notes
    -----
    If subpixel accuracy is good, this should be flat. If it depressed in the
    middle, try using a larger value for feature diameter."""
    f[['x', 'y']].applymap(lambda x: x % 1).hist(ax=ax)
    return ax

@make_axes
def fit(data, fits, inverted_model=False, logx=False, logy=False, ax=None,
        **kwargs):
    data = data.dropna()
    x, y = data.index.values.astype('float64'), data.values
    datalines = plt.plot(x, y, 'o', label=data.name)
    ax = datalines[0].get_axes()
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    if not inverted_model:
        fitlines = ax.plot(fits.index, fits, **kwargs)
    else:
        fitlines = ax.plot(fits.reindex(data.dropna().index),
                           data.dropna(), **kwargs)
    # Restrict plot axis to domain of the data, not domain of the fit.
    xmin = data.index.values[data.index.values > 0].min() if logx \
        else data.index.values.min()
    ax.set_xlim(xmin, data.index.values.max())
    # Match colors of data and corresponding fits.
    [f.set_color(d.get_color()) for d, f in zip(datalines, fitlines)]
    if logx:
        ax.set_xscale('log') # logx kwarg does not always take. Bug?

@make_axes
def plot_principal_axes(img, x_bar, y_bar, cov, ax=None):
    """Plot bars with a length of 2 stddev along the principal axes.

    Attribution
    -----------
    This function is based on a solution by Joe Kington, posted on Stack
    Overflow at http://stackoverflow.com/questions/5869891/
    how-to-calculate-the-axis-of-orientation/5873296#5873296
    """
    def make_lines(eigvals, eigvecs, mean, i):
        """Make lines a length of 2 stddev."""
        std = np.sqrt(eigvals[i])
        vec = 2 * std * eigvecs[:,i] / np.hypot(*eigvecs[:,i])
        x, y = np.vstack((mean-vec, mean, mean+vec)).T
        return x, y
    mean = np.array([x_bar, y_bar])
    eigvals, eigvecs = np.linalg.eigh(cov)
    ax.plot(*make_lines(eigvals, eigvecs, mean, 0), marker='o', color='white')
    ax.plot(*make_lines(eigvals, eigvecs, mean, -1), marker='o', color='red')
    ax.imshow(img)

def examine_jumps(data, jumps):
    fig, axes = plt.subplots(len(jumps), 1)
    for i, jump in enumerate(jumps):
        roi = data.ix[jump-10:jump+10]
        axes[i].plot(roi.index, roi, 'g.')
        axes[i].plot(jump, data[jump], 'ko')
    fig.show()
    fig2, axes2 = plt.subplots(1, 1)
    axes2.plot(data.index, data, 'g.')
    for jump in jumps:
        axes2.plot(jump, data[jump], 'ko')
    fig2.show()

@make_axes
def plot_displacements(t, frame1, frame2, ax=None, **kwargs):
    a = t[t.frame == frame1]
    b = t[t.frame == frame2]
    j= a.set_index('particle')[['x', 'y']].join(
        b.set_index('particle')[['x', 'y']], rsuffix='_b')
    j['dx'] = j.x_b - j.x
    j['dy'] = j.y_b - j.y
    arrow_specs = j[['x', 'y', 'dx', 'dy']].dropna()
    for _, row in arrow_specs.iterrows():
        ax.arrow(*list(row), head_width=4, **kwargs)
    ax.set_xlim(arrow_specs.x.min(), arrow_specs.x.max())
    ax.set_ylim(arrow_specs.y.min(), arrow_specs.y.max())
    return ax
