"""These functions generate handy plots."""
from collections.abc import Iterable
from itertools import tee
from functools import wraps
import warnings
import logging

import numpy as np

try:
    from pims import plot_to_frame, plots_to_frame, normalize
except ImportError:
    plot_to_frame = None
    plots_to_frame = None
    normalize = None


__all__ = ['annotate', 'scatter', 'plot_traj', 'ptraj',
           'annotate3d', 'scatter3d', 'plot_traj3d', 'ptraj3d',
           'plot_displacements', 'subpx_bias', 'mass_size', 'mass_ecc',
           'plot_density_profile']

logger = logging.getLogger(__name__)


def make_axes(func):
    """
    A decorator for plotting functions.
    NORMALLY: Direct the plotting function to the current axes, gca().
              When it's done, make the legend and show that plot.
              (Instant gratificaiton!) The axes have to be 2d or else the
              current figure will be cleared.
    BUT:      If the uses passes axes to plotting function, write on those axes
              and return them. The user has the option to draw a more complex
              plot in multiple steps.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import matplotlib.pyplot as plt
        if kwargs.get('ax') is None:
            kwargs['ax'] = plt.gca()
            # show plot unless the matplotlib backend is headless
            show_plot = (plt.get_backend() != "agg")
        else:
            show_plot = False

        # Delete legend keyword so remaining ones can be passed to plot().
        legend = kwargs.pop('legend', False)

        result = func(*args, **kwargs)

        if legend:
            handles, labels = kwargs['ax'].get_legend_handles_labels()
            if len(labels) > 0:
                kwargs['ax'].legend(handles, labels, loc='best')

        if show_plot:
            plt.show()

        return result
    return wrapper


def make_axes3d(func):
    """
    A decorator for plotting 3d functions.
    NORMALLY: Direct the plotting function to the current axes, gca().
              When it's done, make the legend and show that plot.
              (Instant gratificaiton!) The axes have to be 3d or else the
              current figure will be cleared.
    BUT:      If the uses passes axes to plotting function, write on those axes
              and return them. The user has the option to draw a more complex
              plot in multiple steps.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if kwargs.get('ax') is None:
            if not hasattr(plt.gca(), 'zaxis'):
                plt.figure()  # initialize new Fig when current axis is not 3d
            kwargs['ax'] = plt.gca(projection='3d')
            show_plot = True
        else:
            if not hasattr(plt.gca(), 'zaxis'):
                raise ValueError("The provided axis object is not 3d. Please "
                                 "consult the mplot3d documentation.")
            show_plot = False

        # Delete legend keyword so remaining ones can be passed to plot().
        legend = kwargs.pop('legend', False)

        result = func(*args, **kwargs)

        if legend:
            handles, labels = kwargs['ax'].get_legend_handles_labels()
            if len(labels) > 0:
                kwargs['ax'].legend(handles, labels, loc='best')

        if show_plot:
            plt.show()

        return result
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


def invert_yaxis(ax):
    """Inverts the y-axis of an axis object."""
    bottom, top = ax.get_ylim()
    if top > bottom:
        ax.set_ylim(top, bottom, auto=None)
    return ax


def _plot(ax, coords, pos_columns, **plot_style):
    """ This function wraps Axes.plot to make its call signature the same for
    2D and 3D plotting. The y axis is inverted for 2D plots, but not for 3D
    plots.

    Parameters
    ----------
    ax : Axes object
        The axes object on which the plot will be called
    coords : DataFrame
        DataFrame of coordinates that will be plotted
    pos_columns : list of strings
        List of column names in x, y(, z) order.
    plot_style : keyword arguments
        Keyword arguments passed through to the `Axes.plot(...)` method

    Returns
    -------
    Axes object
    """
    if len(pos_columns) == 3:
        return ax.plot(coords[pos_columns[0]], coords[pos_columns[1]],
                       zs=coords[pos_columns[2]], **plot_style)
    elif len(pos_columns) == 2:
        return ax.plot(coords[pos_columns[0]], coords[pos_columns[1]],
                       **plot_style)


def _set_labels(ax, label_format, pos_columns):
    """This sets axes labels according to a label format and position column
    names. Applicable to 2D and 3D plotting.

    Parameters
    ----------
    ax : Axes object
        The axes object on which the plot will be called
    label_format : string
        Format that is compatible with ''.format (e.g.: '{} px')
    pos_columns : list of strings
        List of column names in x, y(, z) order.

    Returns
    -------
    None
    """
    ax.set_xlabel(label_format.format(pos_columns[0]))
    ax.set_ylabel(label_format.format(pos_columns[1]))
    if hasattr(ax, 'set_zlabel') and len(pos_columns) > 2:
        ax.set_zlabel(label_format.format(pos_columns[2]))


@make_axes
def scatter(centroids, mpp=None, cmap=None, ax=None, pos_columns=None,
            plot_style={}):
    """Scatter plot of all particles.

    Parameters
    ----------
    centroids : DataFrame
        The DataFrame should include time and spatial coordinate columns.
    mpp : float, optional
        Microns per pixel. If omitted, the labels will have units of pixels.
    cmap : colormap, optional
        This is only used in colorby='frame' mode. Default = mpl.cm.winter
    ax : matplotlib axes object, optional
        Defaults to current axes
    pos_columns : list of strings, optional
        Dataframe column names for spatial coordinates. Default is ['x', 'y'].

    Returns
    -------
    Axes object
    
    See Also
    --------
    scatter3d : the 3D equivalent of `scatter`
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if cmap is None:
        cmap = plt.cm.winter
    if pos_columns is None:
        pos_columns = ['x', 'y']
    if len(centroids) == 0:
        raise ValueError("DataFrame of centroids is empty.")
    _plot_style = dict(marker='o', linestyle='none')
    _plot_style.update(**_normalize_kwargs(plot_style, 'line2d'))

    # Axes labels
    if mpp is None:
        _set_labels(ax, '{} [px]', pos_columns)
        mpp = 1.  # for computations of image extent below
    else:
        if mpl.rcParams['text.usetex']:
            _set_labels(ax, r'{} [\textmu m]', pos_columns)
        else:
            _set_labels(ax, r'{} [\xb5m]', pos_columns)

    _plot(ax, centroids, pos_columns, **_plot_style)
    return invert_yaxis(ax)


@make_axes3d
def scatter3d(*args, **kwargs):
    """The 3D equivalent of `scatter`.

    Parameters
    ----------
    centroids : DataFrame
        The DataFrame should include time and spatial coordinate columns.
    mpp : float, optional
        Microns per pixel. If omitted, the labels will have units of pixels.
    cmap : colormap, optional
        This is only used in colorby='frame' mode. Default = mpl.cm.winter
    ax : matplotlib axes object, optional
        Defaults to current axes
    pos_columns : list of strings, optional
        Dataframe column names for spatial coords. Default is ['x', 'y', 'z'].

    Returns
    -------
    Axes object
    
    See Also
    --------
    scatter : the 2D equivalent of `scatter3d`
    """
    if kwargs.get('pos_columns') is None:
        kwargs['pos_columns'] = ['x', 'y', 'z']
    return scatter(*args, **kwargs)


@make_axes
def plot_traj(traj, colorby='particle', mpp=None, label=False,
              superimpose=None, cmap=None, ax=None, t_column=None,
              pos_columns=None, plot_style={}, **kwargs):
    """Plot traces of trajectories for each particle.
    Optionally superimpose it on a frame from the video.

    Parameters
    ----------
    traj : DataFrame
        The DataFrame should include time and spatial coordinate columns.
    colorby : {'particle', 'frame'}, optional
    mpp : float, optional
        Microns per pixel. If omitted, the labels will have units of pixels.
    label : boolean, optional
        Set to True to write particle ID numbers next to trajectories.
    superimpose : ndarray, optional
        Background image, default None
    cmap : colormap, optional
        This is only used in colorby='frame' mode. Default = mpl.cm.winter
    ax : matplotlib axes object, optional
        Defaults to current axes
    t_column : string, optional
        DataFrame column name for time coordinate. Default is 'frame'.
    pos_columns : list of strings, optional
        Dataframe column names for spatial coordinates. Default is ['x', 'y'].
    plot_style : dictionary
        Keyword arguments passed through to the `Axes.plot(...)` command

    Returns
    -------
    Axes object
    
    See Also
    --------
    plot_traj3d : the 3D equivalent of `plot_traj`
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if cmap is None:
        cmap = plt.cm.winter
    if t_column is None:
        t_column = 'frame'
    if pos_columns is None:
        pos_columns = ['x', 'y']
    if len(traj) == 0:
        raise ValueError("DataFrame of trajectories is empty.")
    _plot_style = dict(linewidth=1)
    _plot_style.update(**_normalize_kwargs(plot_style, 'line2d'))

    # Axes labels
    if mpp is None:
        _set_labels(ax, '{} [px]', pos_columns)
        mpp = 1.  # for computations of image extent below
    else:
        if mpl.rcParams['text.usetex']:
            _set_labels(ax, r'{} [\textmu m]', pos_columns)
        else:
            _set_labels(ax, r'{} [\xb5m]', pos_columns)
    # Background image
    if superimpose is not None:
        ax.imshow(superimpose, cmap=plt.cm.gray,
                  origin='lower', interpolation='nearest',
                  vmin=kwargs.get('vmin'), vmax=kwargs.get('vmax'))
        ax.set_xlim(-0.5 * mpp, (superimpose.shape[1] - 0.5) * mpp)
        ax.set_ylim(-0.5 * mpp, (superimpose.shape[0] - 0.5) * mpp)
    # Trajectories
    if colorby == 'particle':
        # Unstack particles into columns.
        unstacked = traj.set_index(['particle', t_column])[pos_columns].unstack()
        for i, trajectory in unstacked.iterrows():
            _plot(ax, mpp*trajectory, pos_columns, **_plot_style)
    if colorby == 'frame':
        # Read http://www.scipy.org/Cookbook/Matplotlib/MulticoloredLine
        x = traj.set_index([t_column, 'particle'])['x'].unstack()
        y = traj.set_index([t_column, 'particle'])['y'].unstack()
        color_numbers = traj[t_column].values/float(traj[t_column].max())
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
        unstacked = traj.set_index([t_column, 'particle'])[pos_columns].unstack()
        first_frame = int(traj[t_column].min())
        coords = unstacked.fillna(method='backfill').stack().loc[first_frame]
        for particle_id, coord in coords.iterrows():
            ax.text(*coord.tolist(), s="%d" % particle_id,
                    horizontalalignment='center',
                    verticalalignment='center')
    return invert_yaxis(ax)

ptraj = plot_traj  # convenience alias

@make_axes3d
def plot_traj3d(*args, **kwargs):
    """The 3D equivalent of `plot_traj`.
    
    Parameters
    ----------
    traj : DataFrame
        The DataFrame should include time and spatial coordinate columns.
    mpp : float, optional
        Microns per pixel. If omitted, the labels will have units of pixels.
    label : boolean, optional
        Set to True to write particle ID numbers next to trajectories.
    superimpose : ndarray, optional
        Background image, default None
    cmap : colormap, optional
        This is only used in colorby='frame' mode. Default = mpl.cm.winter
    ax : matplotlib axes object, optional
        Defaults to current axes
    t_column : string, optional
        DataFrame column name for time coordinate. Default is 'frame'.
    pos_columns : list of strings, optional
        Dataframe column names for spatial coords. Default is ['x', 'y', 'z'].
    plot_style : dictionary
        Keyword arguments passed through to the `Axes.plot(...)` command

    Returns
    -------
    Axes object

    See Also
    --------
    plot_traj : plot 2D trajectories"""

    if kwargs.get('pos_columns') is None:
        kwargs['pos_columns'] = ['x', 'y', 'z']
    if kwargs.get('colorby') == 'frame':
        raise NotImplemented("3d trajectory plots cannot be colored by frame")
    return plot_traj(*args, **kwargs)

ptraj3d = plot_traj3d

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
    -------
    Axes object
    
    See Also
    --------
    annotate3d : The 3D equivalent that returns a scrollable stack.
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
    _imshow_style = dict(origin='lower', interpolation='nearest',
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
    if isinstance(color, str):
        color = [color]
    if not isinstance(split_thresh, Iterable):
        split_thresh = [split_thresh]

    # The parameter image can be an image object or a filename.
    if isinstance(image, str):
        image = plt.imread(image)
    if invert:
        ax.imshow(1-image, **_imshow_style)
    else:
        ax.imshow(image, **_imshow_style)
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(-0.5, image.shape[0] - 0.5)

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
    return invert_yaxis(ax)


def annotate3d(centroids, image, **kwargs):
    """Annotates a 3D image and returns a scrollable stack for display in
    IPython.
    
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
    -------
    pims.Frame object containing a three-dimensional RGBA image

    See Also
    --------
    annotate : annotation of 2D images
    """
    if plots_to_frame is None:
        raise ImportError('annotate3d requires pims 0.3 or later. Please '
                          'install/update pims')

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if image.ndim != 3 and not (image.ndim == 4 and image.shape[-1] in (3, 4)):
        raise ValueError("image has incorrect dimensions. Please input a 3D "
                         "grayscale or RGB(A) image. For 2D image annotation, "
                         "use annotate. Multichannel images can be "
                         "converted to RGB using pims.display.to_rgb.")

    # We want to normalize on the full image and stop imshow from normalizing.
    normalized = (normalize(image) * 255).astype(np.uint8)
    imshow_style = dict(vmin=0, vmax=255)
    if '_imshow_style' in kwargs:
        kwargs['imshow_style'].update(imshow_style)
    else:
        kwargs['imshow_style'] = imshow_style

    max_open_warning = mpl.rcParams['figure.max_open_warning']
    was_interactive = plt.isinteractive()
    try:
        # Suppress warning when many figures are opened
        mpl.rc('figure', max_open_warning=0)
        # Turn off interactive mode (else the closed plots leave emtpy space)
        plt.ioff()

        figures = [None] * len(normalized)
        for i, imageZ in enumerate(normalized):
            fig = plt.figure()
            kwargs['ax'] = fig.gca()
            centroidsZ = centroids[(centroids['z'] > i - 0.5) &
                                   (centroids['z'] < i + 0.5)]
            annotate(centroidsZ, imageZ, **kwargs)
            figures[i] = fig

        result = plots_to_frame(figures, width=512, close_fig=True,
                                bbox_inches='tight')
    finally:
        # put matplotlib back in original state
        if was_interactive:
            plt.ion()
        mpl.rc('figure', max_open_warning=max_open_warning)

    return result


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

def subpx_bias(f, pos_columns=None):
    """Histogram the fractional part of the x and y position.

    Parameters
    ----------
    f : DataFrame
    pos_columns : list of column names, optional

    Notes
    -----
    If subpixel accuracy is good, this should be flat. If it depressed in the
    middle, try using a larger value for feature diameter."""
    if pos_columns is None:
        if 'z' in f:
            pos_columns = ['x', 'y', 'z']
        else:
            pos_columns = ['x', 'y']
    axlist = f[pos_columns].applymap(lambda x: x % 1).hist()
    return axlist

@make_axes
def fit(data, fits, inverted_model=False, logx=False, logy=False, ax=None,
        **kwargs):
    data = data.dropna()
    x, y = data.index.values.astype('float64'), data.values
    datalines = ax.plot(x, y, 'o', label=data.name, **kwargs)
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
def plot_density_profile(f, binsize, blocks=None, mpp=None, fps=None,
                         normed=True, t_column='frame', pos_column='z',
                         ax=None, **kwargs):
    """Plot a histogram showing the density profile in one direction.

    Parameters
    ----------
    f : DataFrame
        positions, including columns 'frame' and 'z'
    binsize : integer
        histogram binsize, if mpp is set, this is in in units of microns
    blocks : integer, optional
        number of density profiles to plot
    mpp : number, optional
        microns per pixel
    fps : number, optional
        frames per second
    normed : boolean
        if true, the histogram is normalized
    t_column : string, default 'frame'
    pos_column : string, default 'z'
    ax : matplotlib axes (optional)
    
    Returns
    -------
    Axes object

    Notes
    -----
    Any other keyword arguments will pass through to matplotlib's `plot`.
    """
    import matplotlib as mpl
    lastframe = f[t_column].max()

    if blocks is None:
        framesperblock = lastframe
    else:
        framesperblock = lastframe // blocks
        if framesperblock == 0:
            raise ValueError('Blocktime too low.')

    if mpp is None:
        ax.set_ylabel('{} [px]'.format(pos_column))
        mpp = 1.  # for computations of image extent below
    else:
        if mpl.rcParams['text.usetex']:
            ax.set_ylabel(r'{} [\textmu m]'.format(pos_column))
        else:
            ax.set_ylabel('{} [\xb5m]'.format(pos_column))

    if normed:
        ax.set_xlabel('N / Ntot')
    else:
        ax.set_xlabel('N')

    if fps is None:
        timeunit = ''
        fps = 1.
    else:
        timeunit = ' s'

    ts = f[t_column].values
    zs = f[pos_column].values * mpp
    bins = np.arange(0, np.max(zs), binsize)
    x_coord = (bins[:-1] + bins[1:])/2
    plotlabel = None

    for first in np.arange(0, lastframe, framesperblock):
        mask = np.logical_and(ts >= first, ts < first + framesperblock)
        count, bins = np.histogram(zs[mask], bins=bins, normed=normed)
        if framesperblock != lastframe:
            plotlabel = '{0:.1f}{2} <= t < {1:.1f}{2}'.format(first / fps,
                           (first + framesperblock) / fps, timeunit)
        ax.plot(count * mpp, x_coord, label=plotlabel, **kwargs)

    return ax


@make_axes
def plot_displacements(t, frame1, frame2, scale=1, ax=None, pos_columns=None,
                       **kwargs):
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
    pos_columns : list of strings, optional
        Dataframe column names for spatial coordinates. Default is ['x', 'y'].
    ax : matplotlib axes (optional)

    Notes
    -----
    Any other keyword arguments will pass through to matplotlib's `annotate`.
    """
    if pos_columns is None:
        pos_columns = ['x', 'y']
    a = t[t.frame == frame1]
    b = t[t.frame == frame2]
    j = (a.set_index('particle')[pos_columns].join(
        b.set_index('particle')[pos_columns], rsuffix='_b'))
    for i in pos_columns:
        j['d' + i] = j[i + '_b'] - j[i]
    arrow_specs = j[pos_columns + ['d' + i for i in pos_columns]].dropna()

    # Arrow defaults
    default_arrow_props = dict(arrowstyle='->', connectionstyle='arc3',
                               linewidth=2)
    kwargs['arrowprops'] = kwargs.get('arrowprops', default_arrow_props)
    for _, row in arrow_specs.iterrows():
        xy = row[pos_columns]  # arrow start
        xytext = xy.values + scale*row[['d' + i for i in pos_columns]].values
        # Use ax.annotate instead of ax.arrow because it is allows more
        # control over arrow style.
        ax.annotate("",
                    xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    **kwargs)
    ax.set_xlim(min(j[pos_columns[0]].min(), j[pos_columns[0] + '_b'].min()),
                max(j[pos_columns[0]].max(), j[pos_columns[0] + '_b'].max()))
    ax.set_ylim(min(j[pos_columns[1]].min(), j[pos_columns[1] + '_b'].min()),
                max(j[pos_columns[1]].max(), j[pos_columns[1] + '_b'].max()))
    _set_labels(ax, '{} [px]', pos_columns)

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
