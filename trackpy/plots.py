"""These functions generate handy plots."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from six.moves import zip
from itertools import tee
from collections import Iterable
from functools import wraps
import warnings

import numpy as np

from .utils import print_update


__all__ = ['annotate', 'annotate3d', 'plot_traj', 'ptraj',
           'plot_displacements', 'subpx_bias', 'mass_size', 'mass_ecc']


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
        import matplotlib.pyplot as plt
        if kwargs.get('ax') is None:
            ax = kwargs['ax'] = plt.gca()
            # Delete legend keyword so remaining ones can be passed to plot().
            legend = kwargs.pop('legend', None)
            result = func(*args, **kwargs)
            if legend or ax.get_legend_handles_labels() != ([], []):
                plt.legend(loc='best')
            plt.show()
            ax.figure.canvas.draw()
            return result
        else:
            return func(*args, **kwargs)
    return wrapper


def make_fig(func):
    """See make_axes."""
    import matplotlib.pyplot as plt

    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'fig' not in kwargs:
            kwargs['fig'] = plt.gcf()
            func(*args, **kwargs)
            plt.show()
        else:
            return func(*args, **kwargs)
    return wrapper


@make_axes
def plot_traj(traj, colorby='particle', mpp=None, label=False,
              superimpose=None, cmap=None, ax=None, t_column=None,
              **kwargs):
    """Plot traces of trajectories for each particle.
    Optionally superimpose it on a frame from the video.

    Parameters
    ----------
    traj : DataFrame including columns x and y
    colorby : {'particle', 'frame'}
    mpp : microns per pixel
        If omitted, the labels will be labeled in units of pixels.
    label : Set to True to write particle ID numbers next to trajectories.
    superimpose : background image, default None
    cmap : This is only used in colorby='frame' mode.
        Default = mpl.cm.winter
    ax : matplotlib axes object, defaults to current axes
    t_column : DataFrame column name
        Default is 'frame'

    Returns
    -------
    None
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if cmap is None:
        cmap = plt.cm.winter
    if t_column is None:
        t_column = 'frame'

    if len(traj) == 0:
        raise ValueError("DataFrame of trajectories is empty.")

    # Axes labels
    if mpp is None:
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')
        mpp = 1.  # for computations of image extent below
    else:
        if mpl.rcParams['text.usetex']:
            ax.set_xlabel(r'x [\textmu m]')
            ax.set_ylabel(r'y [\textmu m]')
        else:
            ax.set_xlabel('x [\xb5m]')
            ax.set_ylabel('y [\xb5m]')
    # Background image
    if superimpose is not None:
        ax.imshow(superimpose, cmap=plt.cm.gray,
                  origin='lower', interpolation='none',
                  vmin=kwargs.get('vmin'), vmax=kwargs.get('vmax'))
        ax.set_xlim(-0.5 * mpp, (superimpose.shape[1] - 0.5) * mpp)
        ax.set_ylim(-0.5 * mpp, (superimpose.shape[0] - 0.5) * mpp)
    # Trajectories
    if colorby == 'particle':
        # Unstack particles into columns.
        unstacked = traj.set_index([t_column, 'particle']).unstack()
        ax.plot(mpp*unstacked['x'], mpp*unstacked['y'], linewidth=1)
    elif colorby == 'frame':
        # Read http://www.scipy.org/Cookbook/Matplotlib/MulticoloredLine
        x = traj.set_index([t_column, 'particle'])['x'].unstack()
        y = traj.set_index([t_column, 'particle'])['y'].unstack()

        print_update("Drawing multicolor lines takes awhile. "
                     "Come back in a minute.")
        for particle in x:
            points = np.array(
                [x[particle].values, y[particle].values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap)
            lc.set_array(x.index)
            ax.add_collection(lc)
        ax.set_xlim(x.apply(np.min).min(), x.apply(np.max).max())
        ax.set_ylim(y.apply(np.min).min(), y.apply(np.max).max())
    else:
        raise ValueError(("unknown value for colorby: {} "
                          "must be one of {{particle, frame}}"
                          ).format(colorby))

    if label:
        unstacked = traj.set_index([t_column, 'particle'])[['x', 'y']].unstack()
        first_frame = int(traj[t_column].min())
        coords = unstacked.fillna(method='backfill').stack().loc[first_frame]
        for particle_id, coord in coords.iterrows():
            plt.text(coord['x'], coord['y'], "%d" % particle_id,
                     horizontalalignment='center',
                     verticalalignment='center')
    ax.invert_yaxis()
    return ax

ptraj = plot_traj  # convenience alias


@make_axes
def annotate(centroids, image, circle_size=None, color=None,
             invert=False, ax=None, split_category=None, split_thresh=None,
             imshow_style={}, plot_style={}):
    """Mark identified features with white circles.

    Parameters
    ----------
    centroids : DataFrame including columns x and y
    image : image array (or string path to image file)
    circle_size : Deprecated.
        This will be removed in a future version of trackpy.
        Use `plot_style={'markersize': ...}` instead.
    color : single matplotlib color or a list of multiple colors
        default None
    invert : If you give a filepath as the image, specify whether to invert
        black and white. Default True.
    ax : matplotlib axes object, defaults to current axes
    split_category : string, parameter to use to split the data into sections
        default None
    split_thresh : single value or list of ints or floats to split
        particles into sections for plotting in multiple colors.
        List items should be ordered by increasing value.
        default None
    imshow_style : dictionary of keyword arguments passed through to
        the `Axes.imshow(...)` command the displays the image
    plot_style : dictionary of keyword arguments passed through to
        the `Axes.plot(...)` command that marks the features

    Returns
    ------
    axes
    """
    import matplotlib.pyplot as plt

    if image.ndim != 2 and not (image.ndim == 3 and image.shape[-1] in (3, 4)):
        raise ValueError("image has incorrect dimensions. Please input a 2D "
                         "grayscale or RGB(A) image. For 3D image annotation, "
                         "use annotate3d. Multichannel images can be "
                         "converted to RGB using pims.display.to_rgb.")

    if circle_size is not None:
        warnings.warn("circle_size will be removed in future version of "
                      "trackpy. Use plot_style={'markersize': ...} instead.")
        if 'marker_size' not in plot_style:
            plot_style['marker_size'] = np.sqrt(circle_size)  # area vs. dia.
        else:
            raise ValueError("passed in both 'marker_size' and 'circle_size'")

    _plot_style = dict(markersize=15, markeredgewidth=2,
                       markerfacecolor='none', markeredgecolor='r',
                       marker='o', linestyle='none')
    _plot_style.update(**_normalize_kwargs(plot_style, 'line2d'))
    _imshow_style = dict(origin='lower', interpolation='none',
                         cmap=plt.cm.gray)
    _imshow_style.update(imshow_style)

    # https://docs.python.org/2/library/itertools.html
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    if color is None:
        color = ['r']
    if isinstance(color, six.string_types):
        color = [color]
    if not isinstance(split_thresh, Iterable):
        split_thresh = [split_thresh]

    # The parameter image can be an image object or a filename.
    if isinstance(image, six.string_types):
        image = plt.imread(image)
    if invert:
        ax.imshow(1-image, **_imshow_style)
    else:
        ax.imshow(image, **_imshow_style)
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(-0.5, image.shape[0] - 0.5)
    ax.invert_yaxis()

    if split_category is None:
        if np.size(color) > 1:
            raise ValueError("multiple colors specified, no split category "
                             "specified")
        _plot_style.update(markeredgecolor=color[0])
        ax.plot(centroids['x'], centroids['y'],
                **_plot_style)
    else:
        if len(color) != len(split_thresh) + 1:
            raise ValueError("number of colors must be number of thresholds "
                             "plus 1")
        low = centroids[split_category] < split_thresh[0]
        _plot_style.update(markeredgecolor=color[0])
        ax.plot(centroids['x'][low], centroids['y'][low],
                **_plot_style)

        for c, (bot, top) in zip(color[1:-1], pairwise(split_thresh)):
            indx = ((centroids[split_category] >= bot) &
                    (centroids[split_category] < top))
            _plot_style.update(markeredgecolor=c)
            ax.plot(centroids['x'][indx], centroids['y'][indx],
                    **_plot_style)

        high = centroids[split_category] >= split_thresh[-1]
        _plot_style.update(markeredgecolor=color[-1])
        ax.plot(centroids['x'][high], centroids['y'][high],
                **_plot_style)
    return ax


def annotate3d(centroids, image, **kwargs):
    """
    An extension of annotate that annotates a 3D image and returns a scrollable
    stack for display in IPython. Parameters: see annotate.
    """
    import matplotlib.pyplot as plt
    from pims.display import scrollable_stack

    if image.ndim != 3 and not (image.ndim == 4 and image.shape[-1] in (3, 4)):
        raise ValueError("image has incorrect dimensions. Please input a 3D "
                         "grayscale or RGB(A) image. For 2D image annotation, "
                         "use annotate. Multichannel images can be "
                         "converted to RGB using pims.display.to_rgb.")

    if kwargs.get('ax') is None:
        kwargs['ax'] = plt.gca()

    for i, imageZ in enumerate(image):
        centroidsZ = centroids[np.logical_and(centroids['z'] > i - 0.5,
                                              centroids['z'] < i + 0.5)]
        ax = annotate(centroidsZ, imageZ, **kwargs)
        fig = ax.get_figure()
        fig.canvas.draw()
        if i == 0:
            w, h = fig.canvas.get_width_height()
            result = np.empty((image.shape[0], h, w, 4))
        plot_rgb = np.fromstring(fig.canvas.tostring_argb(),
                                 dtype=np.uint8)
        plot_rgb = plot_rgb.reshape(h, w, 4)
        result[i] = np.roll(plot_rgb, 3, axis=2)  # argb to rgba
        ax.cla()
    fig.clf()
    return scrollable_stack(result, width=w)


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
    if 'z' in f:
        f[['x','y','z']].applymap(lambda x: x % 1).hist(ax=ax)
    else:
        f[['x', 'y']].applymap(lambda x: x % 1).hist(ax=ax)
    return ax

@make_axes
def fit(data, fits, inverted_model=False, logx=False, logy=False, ax=None,
        **kwargs):
    import matplotlib.pyplot as plt

    data = data.dropna()
    x, y = data.index.values.astype('float64'), data.values
    datalines = plt.plot(x, y, 'o', label=data.name, **kwargs)
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
        ax.set_xscale('log')  # logx kwarg does not always take. Bug?

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
        vec = 2 * std * eigvecs[:, i] / np.hypot(*eigvecs[:, i])
        x, y = np.vstack((mean-vec, mean, mean+vec)).T
        return x, y
    mean = np.array([x_bar, y_bar])
    eigvals, eigvecs = np.linalg.eigh(cov)
    ax.plot(*make_lines(eigvals, eigvecs, mean, 0), marker='o', color='white')
    ax.plot(*make_lines(eigvals, eigvecs, mean, -1), marker='o', color='red')
    ax.imshow(img)

def examine_jumps(data, jumps):
    import matplotlib.pyplot as plt

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
def plot_displacements(t, frame1, frame2, scale=1, ax=None, **kwargs):
    """Plot arrows showing particles displacements between two frames.

    Parameters
    ----------
    t : DataFrame
        trajectories, including columns 'frame' and 'particle'
    frame1 : integer
        frame number
    frame2 : integer
        frame number
    scale : float
        scale factor, if 1 (default) then arrow end is placed at particle
        destination; if any other number arrows are rescaled

    Other Parameters
    ----------------
    ax : matplotlib axes (optional)

    Any other keyword arguments will pass through to matplotlib's `annotate`.
    """
    a = t[t.frame == frame1]
    b = t[t.frame == frame2]
    j = a.set_index('particle')[['x', 'y']].join(
        b.set_index('particle')[['x', 'y']], rsuffix='_b')
    j['dx'] = j.x_b - j.x
    j['dy'] = j.y_b - j.y
    arrow_specs = j[['x', 'y', 'dx', 'dy']].dropna()

    # Arrow defaults
    default_arrow_props = dict(arrowstyle='->', connectionstyle='arc3',
                               linewidth=2)
    kwargs['arrowprops'] = kwargs.get('arrowprops', default_arrow_props)
    for _, row in arrow_specs.iterrows():
        xy = row[['x', 'y']]  # arrow start
        xytext = xy.values + scale*row[['dx', 'dy']].values  # arrow end
        # Use ax.annotate instead of ax.arrow because it is allows more
        # control over arrow style.
        ax.annotate("",
                    xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    **kwargs)
    ax.set_xlim(arrow_specs.x.min(), arrow_specs.x.max())
    ax.set_ylim(arrow_specs.y.min(), arrow_specs.y.max())
    return ax


def _normalize_kwargs(kwargs, kind='patch'):
    """Convert matplotlib keywords from short to long form."""
    # Source:
    # github.com/tritemio/FRETBursts/blob/fit_experim/fretbursts/burst_plot.py
    if kind == 'line2d':
        long_names = dict(c='color', ls='linestyle', lw='linewidth',
                          mec='markeredgecolor', mew='markeredgewidth',
                          mfc='markerfacecolor', ms='markersize',)
    elif kind == 'patch':
        long_names = dict(c='color', ls='linestyle', lw='linewidth',
                          ec='edgecolor', fc='facecolor',)
    for short_name in long_names:
        if short_name in kwargs:
            kwargs[long_names[short_name]] = kwargs.pop(short_name)
    return kwargs
