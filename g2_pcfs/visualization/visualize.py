import numpy as np
import seaborn as sns
from g2_pcfs.visualization import AP_figs_funcs as ap
from matplotlib import pyplot as plt

"""
visualize.py

Include routines here that will visualize parts of your analysis.

This can include taking serialized models and seeing how a trained
model "sees" inputs at each layer, as well as just making figures
for talks and writeups.
"""

def plot_g2(g2_t, g2, label, normalize=False, idx=0, colors=None, cidx=0, figsize=(4 ,4), fig=None):
    if fig is None:
        fig = ap.make_fig(figsize[0], figsize[1])

    if colors is None:
        colors = np.array(sns.color_palette("icefire", 12))

    plt.plot(g2_t, g2, label=label, color=colors[cidx])
    ap.dress_fig(xlabel='$\\tau$ ($\mu$s)', ylabel='$g^{(2)}(\\tau)$', legend=True, tight=False)

    return fig


def plot_fits_vs_raw(x, y_data, y_true, y_fits, labels, scatter=False, true_label=None, input_label=None, normalize=False,
                     idx=0, colors=None, cidxs=np.array([3, 8, 1, 10]), figsize=(4, 4), subplot=True, fontsize=7
                     ):

    ap.set_font_size(fontsize)

    if colors is None:
        colors = np.array(sns.color_palette("icefire", 12))
        colors[10] = [1.0, 0.6, 0.]

    if subplot:
        # figsize[1] = figsize[1] * len(y_fits)
        fig, ax = plt.subplots(nrows=len(y_fits), dpi=150, figsize=figsize, squeeze=False, sharex=True)
        ax = ax.flatten()

        for ctr, y_fit in enumerate(y_fits):
            n = int(normalize)
            y1 = y_data[idx]/(1*(1-n)+max(y_data[idx])*(n))
            y2 = y_fit[idx]/(1*(1-n)+max(y_fit[idx])*(n))
            y3 = y_true[idx]/(1*(1-n)+max(y_true[idx])*(n))
            if scatter:
                ax[ctr].scatter(x, y1, marker='s', color=colors[cidxs[ctr]], s=1, alpha=0.5, label=input_label)
            if not scatter:
                ax[ctr].plot(x, y1, color=colors[cidxs[ctr]], lw=0.5, alpha=0.6, label=input_label)
            ax[ctr].plot(x, y2, color=colors[cidxs[ctr]]*0.6, lw=1, label=labels[ctr])
            ax[ctr].plot(x, y3, '--', color=[0.2, 0.2, 0.2], lw=0.5, label=true_label)
            ax[ctr].set_ylabel('$g^{(2)}(\\tau)$')

        ap.dress_fig(xlabel='$\\tau$ ($\mu$s)', legend=True)

    else:
        for ctr, y_fit in enumerate(y_fits):
            fig, ax = ap.make_fig(figsize[0], figsize[1])
            n = int(normalize)
            y1 = y_data[idx]/(1*(1-n)+max(y_data[idx])*(n))
            y2 = y_fit[idx]/(1*(1-n)+max(y_fit[idx])*(n))
            y3 = y_true[idx]/(1*(1-n)+max(y_true[idx])*(n))
            if scatter:
                ax.scatter(x, y1, marker='s', color=colors[cidxs[ctr]], s=1, alpha=0.5, label=input_label)
            if not scatter:
                ax.plot(x, y1, color=colors[cidxs[ctr]], lw=0.5, alpha=0.6, label=input_label)
            ax.plot(x, y2, color=colors[cidxs[ctr]]*0.6, lw=1, label=labels[ctr])
            ax.plot(x, y3, '--', color=[0.2, 0.2, 0.2], lw=0.5, label=true_label)
            ap.dress_fig(xlabel='$\\tau$ ($\mu$s)', ylabel='$g^{(2)}(\\tau)$', legend=True, tight=False)

    return fig



def plot_fits_vs_true(x, y_data, y_true, y_fits, labels, idx=0, fit_lw=1, cidxs=np.array([3, 8, 1, 10]), figsize=(4, 4)):
    """
    x = x points array for raw data and fit
    y_data = a single y points array of raw data
    y_true = a single y points array of true distribution
    y_fits = list of y point arrays of fitted functions
    labels = labels for each y_fit
    fit_lw = fit linewidth
    cidxs = color indices (to pick nicer colors out of a larger seaborn color array)
    """
    colors = np.array(sns.color_palette("icefire", 12))
    fig = plt.figure(5, dpi=150, figsize=figsize)
    ax3 = plt.subplot(2, 1, 1)
    ax3.scatter(x, y_data[idx], marker='s', color=np.array([1, 1, 1])*0.7, s=1, alpha=0.7)
    ax3.plot(x, y_true[idx], '-s', color='k', ms=1, lw=fit_lw, label='True distribution')
    for i, y in enumerate(y_fits):
        ax3.plot(x, y[idx], color=colors[cidxs[i]], lw=fit_lw, label=labels[i])
    ax3.set_xlabel('$\\tau$ ($\mu$s)')
    ax3.set_ylim([-0.1, max(y_true[idx])*1.5])
    ax3.legend(loc='upper center', ncol=2, framealpha=0.5)

    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(x, y_true[idx], '-s', color='k', ms=1, lw=fit_lw, label='True distribution')
    for i, y in enumerate(y_fits):
        ax4.plot(x, y[idx], color=colors[cidxs[i]], lw=fit_lw, label=labels[i])
    ax4.set_xlabel('$\\tau$ ($\mu$s)')
    ax4.set_xlim([-0.05, 0.05])
    ax4.set_ylim([0, 1.2*y_true[idx][find_nearest(0, x)]])

    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(x, y_true[idx], '-s', color='k', ms=1, lw=fit_lw)
    for i, y in enumerate(y_fits):
        ax5.plot(x, y[idx], color=colors[cidxs[i]], lw=fit_lw, label=labels[i])
    ax5.set_xlabel('$\\tau$ ($\mu$s)')
    ax5.set_xlim([1.05, 0.95])
    ax5.set_ylim([0, 1.2*y_true[idx][find_nearest(+1, x)]])

    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(x, y_true[idx], '-s', color='k', ms=1, lw=fit_lw)
    for i, y in enumerate(y_fits):
        ax6.plot(x, y[idx], color=colors[cidxs[i]], lw=fit_lw, label=labels[i])
    ax6.set_xlabel('$\\tau$ ($\mu$s)')
    ax6.set_xlim([0.25, 0.75])
    # ax6.set_ylim([0.5*y_true[find_nearest(0.5, x)], 1.5*y_true[find_nearest(0.5, x)]])
    ax6.set_ylim([0, 1.5 * y_true[idx][find_nearest(0.5, x)]])

    ap.dress_fig()

    return fig


def plot_mulitple_VAE_output(xdata, yraw, ytest, ytrue, spacer, alpha=0.5, color_fit=None, plot_mean=True,
                             plot_raw_all=False, plot_fit_all=False, colors=None, figsize=(4, 4), labels=None
                             ):

    if color_fit is None:
        color_fit = [0.5, 0.5, 0.5]

    if colors is None:
        colors = np.array(sns.color_palette("icefire", 12))

    if labels is None:
        labels = ['Data', 'True', 'NN fit', 'NN mean fit']

    fig = plt.figure(dpi=150, figsize=figsize)
    for ctr, y in enumerate(ytest):

        # Plot raw data
        if ctr == 0 or plot_raw_all:
            plt.scatter(xdata, yraw + spacer * ctr, marker='s', color=colors[1], s=2, alpha=0.5, label=labels[0])

        # Plot true distribution
        if ctr == 0 or plot_fit_all:
            plt.plot(xdata, ytrue + spacer * ctr, '--', color='k', ms=1, lw=1, label=labels[1])

        # Plot fits
        if ctr == 0:
            plt.plot(xdata, y + spacer * ctr, color=color_fit, alpha=alpha, label=labels[2])
        else:
            plt.plot(xdata, y + spacer * ctr, color=color_fit, alpha=alpha)

    if plot_mean:
        plt.plot(xdata, ytest.mean(0) + spacer * ctr, color=[0.6, 0.2, 0.2], label=labels[3])

    ap.dress_fig(xlabel='$\\tau$ ($\mu$s)', ylabel='$g^{(2)}(\\tau)$', legend=True)


def plot_ensemble_variance(x, g2_ens, g2_true=None, input=None, fill=True, nstd=2, idx=0, cidxs=np.array([3, 8, 1, 10]), figsize=(4, 4), labels=None, colors=None, fontsize=7):
    if labels is None:
        labels = ['$\mu$', '$\mu$ + %d$\sigma$' % nstd, '$\mu$ - %d$\sigma$' % nstd, 'Input']

    if colors is None:
        colors = np.array(sns.color_palette("icefire", 12))

    mean = g2_ens[0][idx]
    var = g2_ens[1][idx]

    ap.set_font_size(fontsize)
    fig, ax = ap.make_fig(figsize)

    if not input is None:
        # plt.scatter(x, input[idx], marker='s', s=2, alpha=0.5, label=labels[-1], color=colors[cidxs[0]])
        # plt.plot(x, input[idx], alpha=0.7, lw=0.5, label=labels[-1], color=colors[cidxs[0]])
        plt.plot(x, input, alpha=0.5, lw=0.5, label=labels[-1], color=colors[cidxs[0]])

    if fill:
        # fill_col = colors[cidxs[0]]
        fill_col = [0, 0, 0]
        plt.fill_between(x, mean + nstd * np.sqrt(var), mean - nstd * np.sqrt(var), linewidth=0.0, color=fill_col, alpha=0.3, label='$\mu$ +/- %d$\sigma$' % nstd)
        # plt.plot(x, mean + nstd * np.sqrt(var), lw=0.2, color=fill_col, alpha=.5)
        # plt.plot(x, mean - nstd * np.sqrt(var), lw=0.2, color=fill_col, alpha=.5)

    if not g2_true is None:
        plt.plot(x, g2_true, '--', label='True', color='k')
    plt.plot(x, mean, label=labels[0], color='k')


    if not fill:
        plt.plot(x, mean + nstd * np.sqrt(var), label=labels[1], color=colors[cidxs[1]])
        plt.plot(x, mean - nstd * np.sqrt(var), label=labels[2], color=colors[cidxs[2]])

    ap.dress_fig(xlabel='$\\tau$ ($\mu$s)', ylabel='$g^{(2)}(\\tau)$', legend=True, tight=False)

    return fig


def plot_ensemble_submodels(x, g2_ens, spacer=0, alpha=1, idx=0, figsize=(4, 4), colors=None, fontsize=7):
    if colors is None:
        colors = np.array(sns.color_palette("icefire", 12))

    ap.set_font_size(fontsize)
    fig, ax = ap.make_fig(figsize)
    for i, s in enumerate(g2_ens[2][:, idx, :]):
        plt.plot(x, s[0:int(len(s)/2)] + i*spacer, color=colors[i], alpha=alpha)

    ap.dress_fig(xlabel='$\\tau$ (ps)', ylabel='$g^{(2)}(\\tau)$', tight=False)

    return fig


def plot_MCMC_variance(x, y, y_MCMC, ppc, y_true=None, cidxs=np.array([3, 8, 1, 10]), figsize=(2, 2), labels=None, colors=None, ppc_alpha=0.5, ppc_color=None, fontsize=7):
    if labels is None:
        # labels = ['$\mu$', '$\mu$ + %d$\sigma$' % nstd, '$\mu$ - %d$\sigma$' % nstd, 'Input']
        labels = ['$\mu$ HMC']

    if ppc_color is None:
        ppc_color = [0.5, 0.5, 0.5]

    if colors is None:
        colors = np.array(sns.color_palette("icefire", 12))

    ap.set_font_size(fontsize)
    fig, ax = ap.make_fig(figsize[0], figsize[1])

    for ctr, p in enumerate(ppc):
        if ctr == 0:
            plt.plot(x, p, lw=0.5, alpha=ppc_alpha, color=ppc_color, label='$\sigma$ HMC')
        else:
            plt.plot(x, p, lw=0.5, alpha=ppc_alpha, color=ppc_color)

    plt.plot(x, y, alpha=0.5, lw=0.5, label='Input', color=colors[cidxs[0]])
    if not y_true is None:
        plt.plot(x, y_true, '--', lw=0.5, label='True', color='k')
    plt.plot(x, y_MCMC, lw=1, label='$\mu$ HMC', color='k')

    ap.dress_fig(xlabel='$\\tau$ ($\mu$s)', ylabel='$g^{(2)}(\\tau)$', legend=True, tight=False, lgd_loc='upper left')

    return fig