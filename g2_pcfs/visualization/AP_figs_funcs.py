import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.ticker import AutoMinorLocator

c      = 299792458  # m/s
π      = 3.141592653589793

# import seaborn as sns
# base_colors = np.array(sns.color_palette("icefire", 12))

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'
# matplotlib.rcParams['axes.labelsize'] = 10
# matplotlib.rcParams['axes.titlesize'] = 10
# matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['axes.linewidth'] = 0.5 #set the value globally


SMALL_SIZE = 9
MEDIUM_SIZE = 9
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def set_font_size(size, lgnd=-2):
    plt.rc('font', size=size)  # controls default text sizes
    plt.rc('axes', titlesize=size)  # fontsize of the axes title
    plt.rc('axes', labelsize=size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=size-lgnd)  # legend fontsize

def find_nearest(value, array):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def dress_fig(ticks=True, legend=True, frameon=False, tight=False, lgnd_cols=1, **kwargs):

    plt.gcf()

    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])
        [ax.set_xlim(kwargs['xlim']) for ax in plt.gcf().axes]

    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
        [ax.set_ylim(kwargs['ylim']) for ax in plt.gcf().axes]

    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
        [ax.set_xlabel(kwargs['xlabel']) for ax in plt.gcf().axes]

    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])
        [ax.set_ylabel(kwargs['ylabel']) for ax in plt.gcf().axes]

    if 'lgd_loc' in kwargs:
        lgd_loc = kwargs['lgd_loc']

    else:
        lgd_loc = 'best'

    if tight:
        plt.tight_layout()

    if legend:
        # plt.legend()
        [ax.legend(loc=lgd_loc, frameon=frameon, handlelength=1, handletextpad=0.5, ncol=lgnd_cols) for ax in plt.gcf().axes] # prop={'size': 8}

    # if 'ticks' in kwargs:
    #     ticks_on = kwargs['ticks']
    if ticks:
        [ax.xaxis.set_minor_locator(AutoMinorLocator()) for ax in plt.gcf().axes]
        [ax.yaxis.set_minor_locator(AutoMinorLocator()) for ax in plt.gcf().axes]

def add_ticks():
    [ax.xaxis.set_minor_locator(AutoMinorLocator()) for ax in plt.gcf().axes]
    [ax.yaxis.set_minor_locator(AutoMinorLocator()) for ax in plt.gcf().axes]

from mpl_toolkits.axes_grid1 import Divider, Size
def make_fig(figsize, dpi=150, fixed=True):
    width = figsize[0]
    height = figsize[1]
    fig = plt.figure(figsize=(width+2, height+2), dpi=dpi)
    if fixed:
        h = [Size.Fixed(1), Size.Fixed(width)]
        v = [Size.Fixed(1), Size.Fixed(height)]
        divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
        ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))
    else:
        ax = fig.add_subplot()
    return fig, ax


def make_subplots(width, height, nrows, ncols, dpi):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width + 2, height + 2), dpi=dpi)
