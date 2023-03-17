import numpy as np
import time
import scipy.io as sio
import matplotlib
import pickle
from matplotlib import pyplot as plt
from numpy.random import randint
import AP_figs_funcs
from MLE_MAP import MLE
from tqdm import tqdm

# Monoexponential (single lifetime) g2 function with 1 side peak, no variable reprate
g2_monoexp = "w[1] + w[3] * (w[0]*np.exp(-np.abs(x)/w[4]) + np.exp(-np.abs(x+w[2])/w[4]) + np.exp(-np.abs(x-w[2])/w[4]))"

# Biexponential (two lifetimes) g2 function with 1 side peak, no variable reprate
# g2_biexp   = "w[0] + w[1] * ( np.exp(-np.abs(x+1)/w[3]) + np.exp(-np.abs(x-1)/w[3]) + w[2]*np.exp(-np.abs(x)/w[3]))" \
#              "     + w[4] * ( np.exp(-np.abs(x+1)/w[5]) + np.exp(-np.abs(x-1)/w[5]) + w[2]*np.exp(-np.abs(x)/w[5]))"

# Biexponential (two lifetimes) g2 function with up to 3 side peaks, variable reprate
g2_biexp   = "w[1] + w[3] * (w[0]*np.exp(-np.abs(x)/w[4]) " \
             "                  + np.exp(-np.abs(x+w[2])/w[4]) + np.exp(-np.abs(x-w[2])/w[4])" \
             "                  + np.exp(-np.abs(x+2*w[2])/w[4]) + np.exp(-np.abs(x-2*w[2])/w[4]) " \
             "                  + np.exp(-np.abs(x+3*w[2])/w[4]) + np.exp(-np.abs(x-3*w[2])/w[4]) )" \
             "     + w[5] * (w[0]*np.exp(-np.abs(x)/w[6])" \
             "                  + np.exp(-np.abs(x+w[2])/w[6]) + np.exp(-np.abs(x-w[2])/w[6])" \
             "                  + np.exp(-np.abs(x+2*w[2])/w[6]) + np.exp(-np.abs(x-2*w[2])/w[6])" \
             "                  + np.exp(-np.abs(x+3*w[2])/w[6]) + np.exp(-np.abs(x-3*w[2])/w[6]) )"

# Parameters for monoexponential g2 with rep rate
                                   # r    y0    rr   a1    t1
theta_monoexp = {'init': np.array([0.5, 0.02, 1.00, 1.0, 4e-2]),
                   'lb': np.array([0.0, 0.00, 0.95, 0.4, 1e-2]),
                   'ub': np.array([1.0, 0.05, 1.05, 1.2, 2e-1])
                }

# Parameters for biexponential g2 with rep rate
                                   # r    y0    rr   a1    t1   a2    t2
theta_biexp   = {'init': np.array([0.5, 0.02, 1.00, 0.7, 4e-2, 0.3, 1e-1]),
                   'lb': np.array([0.0, 0.00, 0.95, 0.0, 1e-2, 0.0, 1e-2]),
                   'ub': np.array([1.0, 0.05, 1.05, 1.0, 2e-1, 1.0, 2e-1])
                }


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def FFTShift(array):
    transformed = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(array))))
    return transformed


def g2_clean(theta, x, function): # function output (clean)
    w = np.array(theta)
    f = eval(function)
    return f


def g2_Poisson(theta, x, T, function): # function output (Poisson)
    w = np.array(theta)
    f = eval(function)
    return np.random.poisson(f*T, size=len(x))


def g2_Poisson_background(theta, x, T, bckgrnd, function): # function output + background (Poisson)
    w = np.array(theta)
    f = eval(function) + bckgrnd
    return np.random.poisson(f*T, size=len(x))


def Poisson_background(x, T, bckgrnd, function): # function output (Poisson)
    c0 = bckgrnd
    f = eval(function)
    return np.random.poisson(f*T, size=len(x))


def g2_Poisson_linear(x, theta, T, function): # function output with Poisson noise in linear spaced bins
    w = np.array(theta)
    f = eval(function)
    return np.random.poisson(f*T, size=len(x))


def g2_Poisson_log(x, theta, T, function): # function output with Poisson noise in log spaced bins
    w = np.array(theta)
    f = eval(function)
    return np.random.poisson(f*x*T, size=len(x))/(x*T)


def load_photon_stream(fname):
    tic = time.time()
    print('Loading photon stream...')
    data = sio.loadmat(fname)
    stream = np.array(data['stream'])
    # print('Stream loaded')
    print('Stream loaded after: %s' % round(time.time() - tic, 3))

    return stream


def intensity_trace(stream, binwidth, plot=False, w=3.5, h=2, threshold=None):
    """
    :param stream: photon stream
    :param binwidth: binwidth (in s)
    :param threshold: threshold under which to kill photons in time bin (in cps)
    :return:
    """
    tbins = np.arange(start=stream[0][1], stop=stream[-1][1], step=binwidth*1e12)
    trace = np.histogram(stream[:, 1], tbins)
    cps   = trace[0]/binwidth

    if plot:
        AP_figs_funcs.make_fig(width=w, height=h, dpi=150)
        plt.plot((tbins[1:]-(tbins[0]))/1e12, cps, lw=0.5)
        AP_figs_funcs.dress_fig(ylabel='cps', xlabel='Time (s)', tight=False)

    if threshold is not None:
        # Loop through and find bins where cps is below threshold; kill photons in that bin
        idx = 0
        logic = np.zeros((len(stream[:, 1])), dtype=bool)
        # for idx in range(len(cps)):
        for idx in tqdm(range(len(cps)), desc="Intensity trace correction"):
            if cps[idx] < threshold:
                l = ((stream[:, 1] >= trace[1][idx]) & (stream[:, 1] <= trace[1][idx+1]))
                logic = logic + l

        # Delete photons in bins that are thresholded
        stream = np.delete(stream, logic, 0)

    # # Generate corrected trace for inspection
    # tbins_c = np.arange(start=stream[0][1], stop=stream[-1][1], step=binwidth * 1e12)
    # trace_c = np.histogram(stream[:, 1], tbins_c)
    # cps_c = trace_c[0] / binwidth

    return stream


def pulsed_g2(photons_data, numbins, tmax, startdetector=1, printing=False, bidirectional=True, tgate=None, offset=200, lastpeak=False, tinf=1e10, reprate=1e6):
    """
    Performs asymmetric correlation on photon stream collected from a Swabian TimeTagger20 (.ttbin files)
    Written by Tara Sverko, Katie Shulenberger, Hendrik Utzat, Timothy Sinclair
    Modified by Andrew Proppe for Python (changed storing temporary histograms as lists and sum them at end. Speed up of ~30% compared ot MATLAB version)

    Inputs:
    photons_data: photon stream(nx3) with first column detector identity 1-4, second column sync pulse times, third column photon arrival times *after* sync pulse
    numbins: total number of bins for g2
    tmax: max time window
    startdetector: designates one detector as start, other as stop (default 1)
    bidirectional: whether or not to generate an asymmetric bidirecitonal g2, or sum forward and backward g2s and have only positive time bins (default True)
    """
    if printing:
        print('Time gating...')

    if offset == 'auto' and tgate != None:
        """
        The automatic offset gets a lifetime histograms from all the photon arrival times relative to the sync pulse
        (third column of the stream data), finds the time at which the lifetime peaks, and subtracts it from all 
        photon arrival times. Removes time delay between sync pulse and detection to make time gating easier 
        """
        stream_tbins = np.linspace(0, 1e6, 10000)
        stream_histo = np.histogram(photons_data[:, 2], bins=stream_tbins)[0]
        real_bins = stream_tbins[0:-1] + np.diff(stream_tbins)[0]
        idx = np.where(stream_histo == max(stream_histo))
        photons_data[:, 2] = photons_data[:, 2] - stream_tbins[idx]

    if tgate != None:
        Logique = np.abs(photons_data[:, 2]) <= tgate  # Look for time differences less than the time gate
        photons_data = np.delete(photons_data, Logique, 0)

    if printing:
        print('Correlating...')
    tic = time.time()
    ii = -1
    g2_temp_forward, g2_temp_backward = [], []
    while 1:
        ii = ii + 1

        batch_shift_down  = photons_data[(ii+1):]
        batch_shift_up    = photons_data[0:(-1-ii)]
        batch_diff        = batch_shift_down - batch_shift_up
        batch_diff_extend = np.append(batch_diff, batch_shift_down[:, 0, None], 1)
        batch_diff_extend = np.append(batch_diff_extend, batch_shift_up[:, 0, None], 1)
        pulse_sorted      = batch_diff_extend

        lastpeak_start = tinf - reprate/2
        lastpeak_end   = tinf + reprate/2

        if lastpeak:
            mask1 = (
                     (np.abs(batch_diff_extend[:, 1]) < 1.5 * tmax)
            )
            mask2 = (
                    (np.abs(batch_diff_extend[:, 1]) >= lastpeak_start) & (np.abs(batch_diff_extend[:, 1]) <= lastpeak_end)
            )
            mask = mask1 + mask2
        else:
            mask = (
                    (np.abs(batch_diff_extend[:, 1]) < 1.5 * tmax)
            )

        pulse_sorted = pulse_sorted[mask]

        pulse_sorted = pulse_sorted[pulse_sorted[:, 0] != 0, :]

        Logic_unfiltered = np.logical_and(np.logical_not(pulse_sorted[:, 3] == 1), np.logical_not(pulse_sorted[:, 4] == 1))

        pulse_all_filtered = pulse_sorted

        try:
            pulse_all_filtered = np.delete(pulse_all_filtered, Logic_unfiltered.squeeze(), 0)
        except:
            pass
        pulse_all_unfiltered = pulse_sorted[Logic_unfiltered]

        Logic3 = pulse_all_filtered[:, 3] == startdetector
        Logic4 = pulse_all_filtered[:, 4] == startdetector
        pulse_bx_forward = pulse_all_filtered[Logic3]
        pulse_bx_backward = pulse_all_filtered[Logic4]

        pulse_bx_forward_time = pulse_bx_forward[:, 2] + [pulse_bx_forward[:, 1]]
        pulse_bx_backward_time = pulse_bx_backward[:, 2] + [pulse_bx_backward[:, 1]]

        if bidirectional:
            time_bins = np.linspace(0, tmax, round(numbins/2)+2)
            if lastpeak:
                time_bins2 = np.linspace(tinf-reprate, tinf+reprate, round(numbins/2)+2)
        if not bidirectional:
            time_bins = np.linspace(0, tmax, numbins+1)
            if lastpeak:
                time_bins2 = np.linspace(tinf-reprate, tinf+reprate, numbins+1)

        if lastpeak:
            time_bins = np.concatenate((time_bins, time_bins2), axis=0)

        g2_temp_forward.append(np.histogram(pulse_bx_forward_time, time_bins)[0])
        g2_temp_backward.append(np.histogram(pulse_bx_backward_time, time_bins)[0])

        if len(pulse_bx_forward) == 0 and len(pulse_bx_backward) == 0 and len(pulse_all_unfiltered) == 0:
            break

    g2_fw   = np.array(sum(g2_temp_forward))
    g2_bw   = np.array(sum(g2_temp_backward))
    g2_tot  = g2_fw + g2_bw

    if bidirectional:
        g2_bw   = np.delete(g2_bw, 0, 0)
        g2      = np.concatenate((np.flip(g2_bw), g2_fw))
        g2      = np.append(g2, g2[0])
        g2      = np.delete(g2, 0)

        g2_t_fw = np.linspace(0, tmax, round(numbins/2)+1)
        if lastpeak:
            g2_t_fw2 = np.linspace(tinf - reprate, tinf + reprate, round(numbins/2)+1)
            g2_t_fw = np.concatenate((g2_t_fw, g2_t_fw2), axis=0)
        g2_t_bw = np.flip(-g2_t_fw[1:])
        g2_t    = np.concatenate((g2_t_bw, g2_t_fw)) # - np.diff(g2_t_fw)[0]

    if not bidirectional:
        g2      = g2_tot
        g2_t    = np.linspace(0, tmax, numbins)

    if printing:
        print('Correlation finished after: %ss' % round(time.time() - tic, 3))

    return g2_fw, g2_bw, g2, g2_t


def solution_pulsed_g2(photons_data, numbins, startdetector=1, printing=False, bidirectional=True, tgate=None, offset=200, rrate=1e6, peak_times=([0, 1e6])):
    """
    Performs asymmetric or symmetric correlation on photon stream collected from a Swabian TimeTagger20 (.ttbin files)
    Written by Andrew Proppe (MIT, 2021), based on original script by Tara Sverko, Katie Shulenberger, Hendrik Utzat, Timothy Sinclair
    - Changed storing temporary histograms as lists and sum them at end. Speed up of ~30% compared ot MATLAB version
    - Added time gating
    - Added option to pick specific time ranges (peaks) to be correlated, without correlating entire stream

    Parameters
    ----------
    photons_data: photon stream(nx3) with first column detector identity 1-4, second column sync pulse times, third column photon arrival times *after* sync pulse
    numbins: total number of bins for g2
    startdetector: designates one detector as start, other as stop (default 1)
    bidirectional: whether or not to generate an asymmetric bidirecitonal g2, or sum forward and backward g2s and have only positive time bins (default True)
    tgate:
        Value of time gate (in ns):
        > first histograms entire photon stream (TCSPC lifetime) to find time offset and subtracts it from all photon time stamps
        > deletes photon events with arrival times less than tgate
        None: time gating is skipped
    offset: time (in ps) to subtract from all photon arrival times. Use 'auto' to get offset from lifetime histogram
    rrate: rep rate used in the experiment (in ps)
    peak_times: times (in ps) for which peaks in the g2 to correlate (e.g. 0 for center peak, rrate*1 for first side peak, rrate*10 for tenth side peak, etc.)
    """
    lifetime_bins = None
    lifetime_histo = None
    tic = time.time()
    if printing:
        print('Time gating...')

    if offset == 'auto' and tgate != None:
        """
        The automatic offset gets a lifetime histograms from all the photon arrival times relative to the sync pulse
        (third column of the stream data), finds the time at which the lifetime peaks, and subtracts it from all 
        photon arrival times. Removes time delay between sync pulse and detection to make time gating easier 
        """
        stream_tbins = np.linspace(0, 1e6, 10000)
        lifetime_histo = np.histogram(photons_data[:, 2], bins=stream_tbins)[0]
        lifetime_bins = stream_tbins[0:-1] + np.diff(stream_tbins)[0]
        idx = np.where(lifetime_histo == max(lifetime_histo))
        photons_data[:, 2] = photons_data[:, 2] - stream_tbins[idx]

    if tgate != None:
        Logique = np.abs(photons_data[:, 2]) <= tgate  # Look for time differences less than the time gate
        photons_data = np.delete(photons_data, Logique, 0)
        if printing:
            print('Time gating finished after: %ss' % round(time.time() - tic, 3))

    if printing:
        print('Correlating...')

    ii = -1
    # g2_temp_forward, g2_temp_backward = [], []
    g2_temp_forward, g2_temp_backward = [[] for i in range(len(peak_times))], [[] for i in range(len(peak_times))]
    peaks = ([], [])
    g2_list = []
    g2_t_list = []
    stop_list = [False for i in range(len(peak_times))]

    while 1:
        ii = ii + 1
        batch_shift_down  = photons_data[(ii+1):]
        batch_shift_up    = photons_data[0:(-1-ii)]
        batch_diff        = batch_shift_down - batch_shift_up
        batch_diff_extend = np.append(batch_diff, batch_shift_down[:, 0, None], 1)
        batch_diff_extend = np.append(batch_diff_extend, batch_shift_up[:, 0, None], 1)
        pulse_sorted_all  = batch_diff_extend

        for ctr, t in enumerate(peak_times):
            if t == 0:
                tstart = 0
                tend   = rrate/2
                mask = (np.abs(batch_diff_extend[:, 1]) < tend)
            else:
                tstart = t - rrate/2
                tend   = t + rrate/2
                mask = (np.abs(batch_diff_extend[:, 1]) >= tstart) & (np.abs(batch_diff_extend[:, 1]) <= tend)

            # pulse_sorted.append(pulse_sorted_all[mask])
            pulse_sorted = pulse_sorted_all[mask]

            pulse_sorted = pulse_sorted[pulse_sorted[:, 0] != 0, :]

            Logic_unfiltered = np.logical_and(np.logical_not(pulse_sorted[:, 3] == 1), np.logical_not(pulse_sorted[:, 4] == 1))

            pulse_all_filtered = pulse_sorted

            try:
                pulse_all_filtered = np.delete(pulse_all_filtered, Logic_unfiltered.squeeze(), 0)
            except:
                pass

            pulse_all_unfiltered = pulse_sorted[Logic_unfiltered]

            Logic3 = pulse_all_filtered[:, 3] == startdetector
            Logic4 = pulse_all_filtered[:, 4] == startdetector
            pulse_bx_forward = pulse_all_filtered[Logic3]
            pulse_bx_backward = pulse_all_filtered[Logic4]

            pulse_bx_forward_time = pulse_bx_forward[:, 2] + [pulse_bx_forward[:, 1]]
            pulse_bx_backward_time = pulse_bx_backward[:, 2] + [pulse_bx_backward[:, 1]]

            if tstart == 0:
                nbins = numbins // 2
            else:
                nbins = numbins

            if bidirectional:
                time_bins = np.linspace(tstart, tend, round(nbins/2)+2)
            if not bidirectional:
                time_bins = np.linspace(tstart, tend, nbins+1)

            g2_temp_forward[ctr].append(np.histogram(pulse_bx_forward_time, time_bins)[0])
            g2_temp_backward[ctr].append(np.histogram(pulse_bx_backward_time, time_bins)[0])

            if len(pulse_bx_forward) == 0 and len(pulse_bx_backward) == 0 and len(pulse_all_unfiltered) == 0:
                stop_list[ctr] = True

        if all(stop_list) == True:
            break

    for ctr, t in enumerate(peak_times):
        if t == 0:
            tstart = 0
        else:
            tstart = t - rrate / 2
        tend = t + rrate / 2

        g2_fw   = np.array(sum(g2_temp_forward[ctr]))
        g2_bw   = np.array(sum(g2_temp_backward[ctr]))
        g2_tot  = g2_fw + g2_bw

        if tstart == 0:
            nbins = numbins // 2
        else:
            nbins = numbins

        if bidirectional:
            g2_bw   = np.delete(g2_bw, 0, 0)
            g2      = np.concatenate((np.flip(g2_bw), g2_fw))
            g2      = np.append(g2, g2[0])
            g2      = np.delete(g2, 0)

            g2_t_fw = np.linspace(tstart, tend, round(nbins/2)+1)
            g2_t_bw = np.flip(-g2_t_fw[1:])
            g2_t    = np.concatenate((g2_t_bw, g2_t_fw)) # - np.diff(g2_t_fw)[0]

        if not bidirectional:
            g2      = g2_tot
            g2_t    = np.linspace(tstart, tend, nbins)

        g2_list.append(g2)
        g2_t_list.append(g2_t)

    if printing:
        print('Correlation finished after: %ss' % round(time.time() - tic, 3))

    return g2_list, g2_t_list, lifetime_bins, lifetime_histo


def peak_area_ratio(g2, g2_t, window_size, rep_rate, baseline_sub=True):
    if baseline_sub:
        g2 = g2 - min(g2)

    centre_start = find_nearest(0 - window_size / 2, g2_t)
    centre_end = find_nearest(0 + window_size / 2, g2_t)
    centre_peak = g2[centre_start:centre_end]

    rightside_start = find_nearest(-rep_rate - window_size / 2, g2_t)
    rightside_end = find_nearest(-rep_rate + window_size / 2, g2_t)
    rightside_peak = g2[rightside_start:rightside_end]

    leftside_start = find_nearest(rep_rate - window_size / 2, g2_t)
    leftside_end = find_nearest(rep_rate + window_size / 2, g2_t)
    leftside_peak = g2[leftside_start:leftside_end]

    peak_height_ratio = sum(centre_peak) / (0.5*(sum(leftside_peak) + sum(rightside_peak)))

    return peak_height_ratio


class g2_SimulatedDataset(object):
    """
    Generates a dataset of simulated g2 functions (g2_true) based off input parameters (theta) that are Poisson sampled to add noise (y)
    """
    def __init__(self):
        self.g2_t = None
        self.counts = None
        self.g2 = None
        self.g2_norm = None
        self.g2_true = None
        self.g2_true_norm = None
        self.theta = None

    def generate(self, function, theta, nbins, trange, it, ndata, normalize=None, submin=False, random_vars=[]):
        """
        Inputs:
        function: g2 function using w arrays as variables (e.g. funk = "w[0] + w[1] * ( np.exp(-np.abs(x+1)/w[3]) ...)
        theta = array of w parameters used to generate the g2
        nbins = number of x points (time bins)
        trange = min and max of time range
        it = 'integration max', changes overall amplitude of g2. Higher it = less noise (cleaner function)
        ndata = number of g2s to generate
        Normalize = 'test': divide each g2_test and g2_true by max of g2_test
                    'true': divide each g2_test and g2_true by max of g2_true
                    'dual': divide each g2_test and g2_true by their own maxima (non-relative normalization)
                    None: no normalization. g2_norm and g2_true_norm are still generated by normalizing relative to g2_true
        submin = subtract minimum from all g2_true's and y's
        random_r = for each g2, generate a random number between 0 and 1 for the r parameter
        """
        self.g2_t = np.linspace(trange[0], trange[1], nbins)
        self.g2 = []
        self.g2_norm = []
        self.g2_true = []
        self.g2_true_norm = []
        self.theta = []
        self.counts = np.zeros(ndata)

        for n in range(ndata):
            theta_temp = theta['init'].copy()
            for r in random_vars:
                theta_temp[r] = np.random.uniform(theta['lb'][r], theta['ub'][r])

            g2_true = g2_clean(theta=theta_temp, x=self.g2_t, function=function) # Generate clean function
            g2_true /= max(g2_true) # Normalize to 1 (ignore amplitude parameter)
            g2_true *= it # Scale by 'integration time'

            g2_test = np.random.poisson(g2_true, size=len(g2_true))
            self.counts[n] = sum(g2_test)

            if submin:
                g2_test = g2_test - min(g2_test)
                g2_true = g2_true - min(g2_true)

            if normalize == 'test':
                maxVal = max(g2_test)
                g2_test = g2_test / maxVal
                g2_true = g2_true / maxVal

            elif normalize == 'true':
                maxVal = max(g2_true)
                g2_test = g2_test / maxVal
                g2_true = g2_true / maxVal

            elif normalize == 'dual':
                maxVal = max(g2_true)
                g2_test = g2_test / max(g2_test)
                g2_true = g2_true / max(g2_true)

            else:
                maxVal = max(g2_true)

            self.g2.append(g2_test)
            self.g2_norm.append(g2_test/maxVal)
            self.g2_true.append(g2_true)
            self.g2_true_norm.append(g2_true/maxVal)
            self.theta.append(theta_temp)


class g2_ExperimentalDataset(object):
    def __init__(self, fname):
        self.g2_t         = None
        self.counts       = None
        self.g2           = None
        self.g2_norm      = None
        self.g2_full      = None # g2 from full photon stream
        self.g2_true      = None # Mono or biexponential fit to g2_full
        self.g2_true_norm = None
        self.theta        = None
        self.parr_true    = None
        self.max_cts      = None
        self.mle          = None
        self.numbins      = None
        self.tmax         = None
        self.fname        = fname
        self.stream       = load_photon_stream(self.fname)

    def get_g2_true(self, function, theta, numbins=301, tmax=1.5e6, bidirectional=False):
        """
        Performs g2 correlation across entire photon stream to get g2_full, which is then fit using MLE to get the true g2 (g2_true),
        true theta (theta), the true peak-area-ratio (
        """
        print('Correlating full stream to get g2 true..')
        g2_fw, g2_bw, g2_full, g2_t = pulsed_g2(photons_data=self.stream, numbins=numbins, tmax=tmax, bidirectional=bidirectional) # Perform full correlation

        self.max_cts = sum(g2_full)
        g2_t         = g2_t/1e6
        self.g2_t    = g2_t
        self.g2_full = g2_full

        theta['init'][1] *= self.max_cts
        theta['init'][3] *= self.max_cts
        theta['init'][5] *= self.max_cts
        theta['ub'][1]   *= self.max_cts
        theta['ub'][3]   *= self.max_cts
        theta['ub'][5]   *= self.max_cts

        print('Using Maximum Likelihood Estimation to get fit to g2 true..')
        self.mle = MLE(
            x=self.g2_t,
            y=self.g2_full,
            nruns=10,
            nvars=len(theta['init']),
            fit_function=function,
            obj_functions=['Gaussian'],
            guess=theta['init'],
            randomizer='bounded',
            bounds=([theta['lb'], theta['ub']]),
            printing=False
        )

        self.g2_true  = self.mle.fit_L
        self.theta    = self.mle.theta_L
        self.par_true = peak_area_ratio(self.g2_true, self.g2_t, window_size=0.5, rep_rate=1)
        self.numbins  = numbins
        self.tmax     = tmax
        self.bidirectional = bidirectional

    def generate(self, count_target, ndata, target_on=True, normalize=True):
        if self.numbins is None or self.tmax is None:
            print('Run .g2_true() first to specify numbins and tmax')
            pass
        fraction_to_corr = count_target / self.max_cts

        self.g2           = []
        self.g2_norm      = []
        self.g2_true      = []
        self.g2_true_norm = []
        self.theta        = []

        print('Correlating substreams to get g2s..')
        for n in range(ndata):
            ctr, stop = 0, 0
            while stop == 0:
                a = len(self.stream)
                b = round(a * fraction_to_corr)
                c = randint(0, a - b)
                print('Attempt %s for g2 %s' % (ctr, n), end='\r')
                g2_fw, g2_bw, g2, g2_t = pulsed_g2(photons_data=self.stream[c:c + b], numbins=self.numbins, tmax=self.tmax, bidirectional=self.bidirectional)
                if 0.95 * count_target < sum(g2) < 1.05 * count_target:
                    stop = 1
                ctr += 1
                if not target_on:
                    stop = 1

            g2_true = self.mle.fit_L.copy()
            g2_true = g2_true * count_target / self.max_cts

            maxVal = max(g2)

            self.g2_norm.append(g2/maxVal)
            self.g2_true_norm.append(g2_true/maxVal)

            if normalize:
                g2      = g2/maxVal
                g2_true = g2_true/maxVal

            self.g2.append(g2)
            self.g2_true.append(g2_true)
            self.theta.append(self.mle.theta_L)

        self.ndata        = ndata
        self.count_target = count_target
        self.name         = self.fname.split('/')[-1].split('.')[0]+'_counts'+str(count_target)+'_ndata'+str(ndata)


    def save(self, name, path='/Volumes/GoogleDrive/My Drive/python/g2-vae/data/expt'):
        # name = self.fname.split('/')[-1].split('.')[0]
        pathfull = path+'/'+name+'.pickle'

        stream_temp = self.stream # temporarily store stream outside of self
        self.stream = []          # don't save stream in self to avoid large file sizes
        with open(pathfull, 'wb') as f:
            pickle.dump(self, f)

        self.stream = stream_temp # restore stream

    @classmethod
    def load(cls, fname, path='/Volumes/GoogleDrive/My Drive/python/g2-vae/data/expt'):
        name = fname.split('/')[-1].split('.')[0]
        pathfull = path+'/'+name+'.pickle'
        with open(pathfull, 'rb') as f:
            return pickle.load(f)


# def generate_simulated_g2_dataset(function, theta, nbins, trange, it, ndata, normalize=True, submin=False, random_r=False):
#     x       = np.linspace(trange[0], trange[1], nbins)
#     if random_r:
#         theta[2] = np.random.uniform()
#     y_true  = g2_clean(theta=theta, x=x, function=function)*it # Generate clean function
#     dataset = ([], [])
#     counts  = np.zeros(ndata)
#
#     for n in range(ndata):
#         y = np.random.poisson(y_true, size=len(y_true))
#         counts[n] = sum(y)
#         if submin:
#             y = y - min(y)
#         if normalize:
#             y = y / max(y_true)
#         dataset[0].append(y)
#         dataset[1].append(theta)
#
#     if submin:
#         y_true = y_true - min(y_true)
#     if normalize:
#         y_true = y_true/max(y_true)
#
#     counts_avg = counts.mean()
#     counts_std = counts.std()
#
#     return x, y, y_true, dataset, counts, counts_avg, counts_std


# Monoexponential (single lifetime) g2 function with 1 side peak, no variable reprate
# monoexp_g2 = "w[0] + w[1] * ( np.exp(-np.abs(x+1)/w[3]) + np.exp(-np.abs(x-1)/w[3]) + w[2]*np.exp(-np.abs(x)/w[3]))"
# monoexp_g2 = "w[1] * ( np.exp(-np.abs(x+1)/w[2]) + np.exp(-np.abs(x-1)/w[2]) + w[0]*np.exp(-np.abs(x)/w[2])) + w[3]"
#
# # Biexponential (two lifetimes) g2 function with 1 side peak, no variable reprate
# biexp_g2   = "  w[1] * ( np.exp(-np.abs(x+1)/w[2]) + np.exp(-np.abs(x-1)/w[2]) + w[0]*np.exp(-np.abs(x)/w[2])) " \
#              "+ w[3] * ( np.exp(-np.abs(x+1)/w[4]) + np.exp(-np.abs(x-1)/w[4]) + w[0]*np.exp(-np.abs(x)/w[4])) " \
#              "+ w[5] "  # fixed rep rate, two lifetimes

# # Parameters for monoexponential g2
#                                    # r    a1    t1    y0
# theta_monoexp = {'init': np.array([0.5, 1.0, 0.040, 0.02]),
#                  'lb':   np.array([0.0, 0.4, 0.001, 0.00]),
#                  'ub':   np.array([1.0, 1.2, 0.200, 0.20])
#                 }
# # Parameters for biexponential g2
#
#                                    # r   a1   a2     t1     t2    y0
# theta_biexp   = {'init': np.array([0.5, 0.8, 0.2, 0.040, 0.060, 0.02]),
#                  'lb':   np.array([0.0, 0.4, 0.0, 0.001, 0.001, 0.00]),
#                  'ub':   np.array([1.0, 1.2, 1.2, 0.200, 0.200, 0.20])
#                 }

# Old working version of pulsed g2
# def pulsed_g2(photons_data, numbins, tmax, startdetector=1, printing=False, bidirectional=True, tgate=None, offset=200, lastpeak=False, tinf=1e10, reprate=1e6):
#     """
#     Performs asymmetric correlation on photon stream collected from a Swabian TimeTagger20 (.ttbin files)
#     Written by Tara Sverko, Katie Shulenberger, Hendrik Utzat, Timothy Sinclair
#     Modified by Andrew Proppe for Python (changed storing temporary histograms as lists and sum them at end. Speed up of ~30% compared ot MATLAB version)
#
#     Inputs:
#     photons_data: photon stream(nx3) with first column detector identity 1-4, second column sync pulse times, third column photon arrival times *after* sync pulse
#     numbins: total number of bins for g2
#     tmax: max time window
#     startdetector: designates one detector as start, other as stop (default 1)
#     bidirectional: whether or not to generate an asymmetric bidirecitonal g2, or sum forward and backward g2s and have only positive time bins (default True)
#     """
#     if printing:
#         print('Time gating...')
#
#     if offset == 'auto' and tgate != None:
#         """
#         The automatic offset gets a lifetime histograms from all the photon arrival times relative to the sync pulse
#         (third column of the stream data), finds the time at which the lifetime peaks, and subtracts it from all
#         photon arrival times. Removes time delay between sync pulse and detection to make time gating easier
#         """
#         stream_tbins = np.linspace(0, 1e6, 10000)
#         stream_histo = np.histogram(stream[:, 2], bins=stream_tbins)[0]
#         real_bins = stream_tbins[0:-1] + np.diff(stream_tbins)[0]
#         idx = np.where(stream_histo==max(stream_histo))
#         photons_data[:, 2] = photons_data[:, 2] - stream_tbins[idx]
#
#     if tgate != None:
#         # diff_array = np.diff(photons_data, axis=0)
#         # Logic6 = diff_array[:, 1] == 0  # Look for zero time delay between sync pulses, meaning one sync pulse triggered two photon events
#         # Logic7 = np.abs(diff_array[:, 2]) <= tgate  # Look for time differences less than the time gate
#         # Logique = np.logical_and(Logic6, Logic7)
#         # Logique = np.append(Logique, False)
#         Logique = np.abs(photons_data[:, 2]) <= tgate  # Look for time differences less than the time gate
#         photons_data = np.delete(photons_data, Logique, 0)
#
#     # Logic6 = np.diff(photons_data[:, 1]) == 0
#     # Logic6 = np.append(Logic6, False)
#     # isame = np.where(Logic6 == True)
#
#     if printing:
#         print('Correlating...')
#     tic = time.time()
#     ii = -1
#     g2_temp_forward, g2_temp_backward = [], []
#     while 1:
#         ii = ii + 1
#
#         batch_shift_down  = photons_data[(ii+1):]
#         batch_shift_up    = photons_data[0:(-1-ii)]
#         batch_diff        = batch_shift_down - batch_shift_up
#         batch_diff_extend = np.append(batch_diff, batch_shift_down[:, 0, None], 1)
#         batch_diff_extend = np.append(batch_diff_extend, batch_shift_up[:, 0, None], 1)
#         pulse_sorted      = batch_diff_extend
#
#         lastpeak_start = tinf - reprate
#         lastpeak_end   = tinf + reprate
#         Logic1 = np.abs(batch_diff_extend[:, 1]) > 1.5*tmax
#         if lastpeak:
#             Logic1a = np.abs(batch_diff_extend[:, 1]) < lastpeak_start
#             Logic1b = np.abs(batch_diff_extend[:, 1]) > lastpeak_end
#             Logic = Logic1 + Logic1a + Logic1b
#         else:
#             Logic = Logic1
#         # Logic = np.logical_or(Logic1, Logic2)
#         pulse_sorted = np.delete(pulse_sorted, Logic, 0)
#
#         pulse_sorted = pulse_sorted[pulse_sorted[:, 0] != 0, :]
#
#         Logic_unfiltered = np.logical_and(np.logical_not(pulse_sorted[:, 3] == 1), np.logical_not(pulse_sorted[:, 4] == 1))
#
#         pulse_all_filtered = pulse_sorted
#         try:
#             pulse_all_filtered = np.delete(pulse_all_filtered, Logic_unfiltered.squeeze(), 0)
#         except:
#             pass
#         pulse_all_unfiltered = pulse_sorted[Logic_unfiltered]
#
#         Logic3 = pulse_all_filtered[:, 3] == startdetector
#         Logic4 = pulse_all_filtered[:, 4] == startdetector
#         pulse_bx_forward = pulse_all_filtered[Logic3]
#         pulse_bx_backward = pulse_all_filtered[Logic4]
#
#         pulse_bx_forward_time = pulse_bx_forward[:, 2] + [pulse_bx_forward[:, 1]]
#         pulse_bx_backward_time = pulse_bx_backward[:, 2] + [pulse_bx_backward[:, 1]]
#
#         if bidirectional:
#             time_bins = np.linspace(0, tmax, round(numbins/2)+2)
#             if lastpeak:
#                 time_bins2 = np.linspace(tinf-reprate, tinf+reprate, round(numbins/2)+2)
#         if not bidirectional:
#             time_bins = np.linspace(0, tmax, numbins+1)
#             if lastpeak:
#                 time_bins2 = np.linspace(tinf-reprate, tinf+reprate, numbins+1)
#
#         if lastpeak:
#             time_bins = np.concatenate((time_bins, time_bins2), axis=0)
#
#         g2_temp_forward.append(np.histogram(pulse_bx_forward_time, time_bins)[0])
#         g2_temp_backward.append(np.histogram(pulse_bx_backward_time, time_bins)[0])
#
#         if len(pulse_bx_forward) == 0 and len(pulse_bx_backward) == 0 and len(pulse_all_unfiltered) == 0:
#             break
#
#     g2_fw   = np.array(sum(g2_temp_forward))
#     g2_bw   = np.array(sum(g2_temp_backward))
#     g2_tot  = g2_fw + g2_bw
#
#     if bidirectional:
#         g2_bw   = np.delete(g2_bw, 0, 0)
#         g2      = np.concatenate((np.flip(g2_bw), g2_fw))
#         g2      = np.append(g2, g2[0])
#         g2      = np.delete(g2, 0)
#
#         g2_t_fw = np.linspace(0, tmax, round(numbins/2)+1)
#         if lastpeak:
#             g2_t_fw2 = np.linspace(tinf - reprate, tinf + reprate, round(numbins/2)+1)
#             g2_t_fw = np.concatenate((g2_t_fw, g2_t_fw2), axis=0)
#         g2_t_bw = np.flip(-g2_t_fw[1:])
#         g2_t    = np.concatenate((g2_t_bw, g2_t_fw)) # - np.diff(g2_t_fw)[0]
#
#     if not bidirectional:
#         g2      = g2_tot
#         g2_t    = np.linspace(0, tmax, numbins)
#
#     if printing:
#         print('Correlation finished after: %ss' % round(time.time() - tic, 3))
#
#     return g2_fw, g2_bw, g2, g2_t