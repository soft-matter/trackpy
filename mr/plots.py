import matplotlib.pyplot as plt
import logging
from motion import *

logger = logging.getLogger(__name__)

def plot_drift(x, label=''):
    """Plot ensemble drift. To compare drifts of subsets, call multiple times
    with finish=False for all but the last call."""
    plt.plot(x[:, 0], x[:, 1], '-', label=label + ' X')
    plt.plot(x[:, 0], x[:, 2], '-', label=label + ' Y')
    plt.xlabel('time [frames]')
    plt.ylabel('drift [px]')
    plt.legend(loc='best')
    plt.show()

def plot_traj(probes, mpp, superimpose=None):
    """Plot traces of trajemr.locate('/media/Frames/T62S1/T62S1F00001.png', 9)
ctories for each probe.
    Optionally superimpose it on a fram from the video."""
    probes = cast_probes(probes)
    if superimpose:
        image = 1-plt.imread(superimpose)
        plt.imshow(image, cmap=cm.gray)
        plt.xlim(0, image.shape[1])
        plt.ylim(0, image.shape[0])
        logger.info("Using units of px, not microns.")
        mpp = 1
        plt.xlabel('x [px]')
        plt.ylabel('y [px]')
    else:
        plt.xlabel('x [um]')
        plt.ylabel('y [um]')
    for traj in probes:
        plt.plot(mpp*traj[:, 1], mpp*traj[:, 2])
    plt.show()

def plot_msd(probes, mpp, fps, max_interval=None, defer=False):
    "Plot MSD for each probe individually."
    logger.info("%.3f microns per pixel, %d fps", mpp, fps)
    probes = cast_probes(probes)
    msds = [msd(traj, mpp, fps, max_interval, detail=False) \
            for traj in probes] 
    for counter, m in enumerate(msds):
        # Label only one instance for the plot legend.
        if counter == 0:
            plt.loglog(m[:, 0], m[:, 1], 'k.-', alpha=0.3,
                       label='individual probe MSDs')
        else:
            plt.loglog(m[:, 0], m[:, 1], 'k.-', alpha=0.3)
    if not defer:
        _config_msd_plot()
        plt.show()

def plot_emsd(probes, mpp, fps, max_interval=None, powerlaw=True, defer=False):
    "Plot ensemble MSDs for probes."
    logger.info("%.3f microns per pixel, %d fps", mpp, fps)
    m = ensemble_msd(probes, mpp, fps, max_interval)
    plt.loglog(m[:, 0], m[:, 1], 'ro-', linewidth=3, label='ensemble MSD')
    if powerlaw:
        power, coeff = fit_powerlaw(m)
        plt.loglog(m[:, 0], coeff*m[:, 0]**power, '-', color='#019AD2', linewidth=2,
                   label=_powerlaw_label(power, coeff))
    if not defer:
        _config_msd_plot()
        plt.show()

def plot_bimodal_msd(probes, mpp, fps, max_interval=None):
    probes = cast_probes(probes)
    upper_branch, lower_branch, middle_branch = split_branches(probes)
    plot_msd(upper_branch, mpp, fps, max_interval, defer=True)
    plot_emsd(upper_branch, mpp, fps, max_interval, powerlaw=True, defer=True)
    plot_msd(middle_branch, mpp, fps, max_interval, powerlaw=False, defer=True)
    plot_msd(lower_branch, mpp, fps, max_interval, defer=True)
    plot_emsd(lower_branch, mpp, fps, max_interval, powerlaw=True, defer=True)

def _config_msd_plot():
    # Label ticks with plain numbers, not scientific notation:
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.ylim(0.01, 100)
    logger.info('Limits of y range are manually set to %f - %f.', *plt.ylim())
    plt.xlabel('lag time [s]')
    plt.ylabel('msd [um$^2$]')
    plt.legend(loc='upper left')
    plt.show()

def _powerlaw_label(power, coeff):
    """Return a string suitable for a legend label, including power
    and D if motion is diffusive, but only power if it is subdiffusive."""
    DIFFUSIVE_THRESHOLD = 0.90
    label = 'power law fit\nn=' + '{:.2f}'.format(power)
    if power >= DIFFUSIVE_THRESHOLD:
        label += '  D=' + '{:.3f}'.format(coeff/4) + ' um$^2$/s'
    return label
    
    
