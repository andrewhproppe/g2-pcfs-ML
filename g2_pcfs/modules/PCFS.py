'''
Python for analysis of photon resolved PCFS file as generated from the Labview instrument control softeware by Hendrik Utzat V3.0.

Adapted from PCFS.m V4.0 @ Hendrik Utzat 2017.

Weiwei Sun, July, 2019

Updated by Andrew Proppe, 2020-2022
'''
import time as timing
import os, re, glob
import g2_pcfs.modules.photons as ph
import seaborn as sns
import pandas as pd
from math import ceil, floor
from matplotlib.animation import FuncAnimation
from scipy import interpolate
from scipy.special import erf
from g2_pcfs.modules.conversions import *
from sklearn.preprocessing import normalize
from scipy.signal import correlate
from g2_pcfs.visualization.AP_figs_funcs import find_nearest as fnear
from g2_pcfs.visualization.AP_figs_funcs import *
from g2_pcfs.modules.MLE_MAP import MLE, loss_AutoCorr_MSE, loss_interferogram_wiener, loss_AutoCorrFFT_MSE, loss_AutoCorrFFT, autocorrelate_peaks, lossL, function_clean

def eformat(f, prec, exp_digits):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d"%(mantissa, exp_digits+1, int(exp))

class PCFS:

    '''
    Called when creating a PCFS object. This function saves the fundamental arguments as object properties.
    folder_path is the full path to the folder containin the photon stream data and meta data of the PCFS run. NOTE: '\' needs to be replaced to '\\' or '/' and the not ending with '/', i.e., 'D/Downloads/PCFS/DotA'.
    memory_limit is the maximum memory to read metadata at once, set to default 1 MB.
    '''
    def __init__(self, folder_path, memory_limit=1, header_lines_pcfslog=5):
        # property
        self.division_factor = None
        self.lags = None
        self.lag_bin_edges = None
        self.cross_correlations = None
        self.auto_correlations = None
        self.tau = None
        self.PCFS_interferogram = None
        # self.interferogram['Blinking corrected interferogram'] = None
        self.interferogram = {}
        self.spectral_correlation = {}
        self.Fourier = {}
        self.memory_limit = memory_limit

        # extract files information
        self.path_str = folder_path
        self.PCFS_ID = os.path.split(folder_path)[1] # extract the folder name, i.e., 'DotA'

        # Identify stage position file; opened and read during 'read_photon_streams'
        os.chdir(folder_path)
        file_pos = glob.glob('*.pos')
        if len(file_pos) == 0:
            print('.pos file not found!')
        self.file_pos = file_pos[0] # with extension

        # Identify and read PCFS log file
        file_pcfslog = glob.glob('*.pcfslog')
        if len(file_pcfslog) == 0:
            print('.pcfslog file not found!')

        self.file_pcfslog = file_pcfslog[0] # with extension

        # read in the metadata of the .pcfslog file and store it as property
        with open(self.file_pcfslog) as f:
            lines_skip_header = f.readlines()[header_lines_pcfslog:]
        self.pcfslog = {}
        for lines in lines_skip_header:
            lines_split = lines.split('=')
            if len(lines_split) == 2:
                self.pcfslog[lines_split[0]] = float(lines_split[1])

        self.photons = {}
        tic = timing.time()
        toc = timing.time()
        # print('Total time elapsed to create PCFS class is %4f s' % (toc - tic))

    def read_photon_streams(self):
        """
        Separate function to read photon streams, which allows the same dot to be analyzed while the experiment is running without
        having to reconvert .streams to .photons, and without having to redo correlations that are already performed and stored.
        """
        self.file_stream = sorted([f.replace('.stream', '') for f in glob.glob('*.stream')]) # without extension
        self.stage_positions = np.loadtxt(self.file_pos)
        self.stage_positions = np.unique(self.stage_positions) # Delete duplicates if any were printed to the file by mistake

        if len(self.stage_positions) != self.pcfslog['N_steps']:
            print("PCFS scan appears to be incomplete; omitting the last .stream file and stage position")
            self.file_stream.remove(self.file_stream[-1]) # for partially completed runs, this ignores the last stage position stream file
            self.stage_positions = self.stage_positions[:len(self.file_stream)]
            self.optical_delay = 2*self.stage_positions/(299792458)/1000*1e12
            # self.stage_positions = np.delete(self.stage_positions, -1) # The stage positions need to be read in on this function call, or else it will continually delete the last stage position whenever the same dot is reloaded.

        self.get_file_photons() # get all the photons files in the current directory

        # create photons object for the photon stream at each interferometer path length difference.
        for f in self.file_stream:
            try:
                self.photons[f]
            except:
                self.photons[f] = ph.photons(f+'.stream', self.memory_limit)
                print('Converted stream "'+f+'" to photons', end='\r')

    '''
    ============================================================================================
    Get and parse photon stream / get correlation functions
    '''

    '''
    This function gets all the photon files in the current directory.
    '''
    def get_file_photons(self):
        self.file_photons = [f.replace('.photons','') for f in glob.glob('*.photons')] # without extension
        self.file_photons = sorted(self.file_photons)

    '''
    This function gets all photon stream data.
    '''
    def get_photons_all(self):
        time_start = timing.time()
        self.get_file_photons()
        for f in self.file_stream:
            if f not in self.file_photons:
                print(f)
                self.photons[f].get_photon_records(memory_limit=self.memory_limit)
        time_end = timing.time()
        # self.photons = sorted(self.photons)
        print('Total time elapsed to get all photons is %4f s' % (time_end - time_start))



    '''
    This function gets the sum signal of the two detectors for all photon arrival files.
    '''
    def get_sum_signal_all(self):
        time_start = timing.time()
        self.get_file_photons()
        for f in self.file_photons:

            if 'sum' not in f and ('sum_signal_' + f) not in self.file_photons :
                self.photons[f].write_photons_to_one_channel(f, 'sum_signal_'+f)
        time_end= timing.time()
        print('Total time elapsed to get sum signal of photons is %4f s' % (time_end - time_start))

    @staticmethod
    def make_tau(time_bounds, lag_precision):
        start_time, stop_time = time_bounds

        '''create log 2 spaced lags'''
        cascade_end = int(np.log2(stop_time)) # cascades are collections of lags  with equal bin spacing 2^cascade
        nper_cascade =  lag_precision # number of equal
        a = np.array([2**i for i in range(1, cascade_end+1)])
        b = np.ones(nper_cascade)
        division_factor = np.kron(a,b)
        lag_bin_edges = np.cumsum(division_factor/2)
        lags = (lag_bin_edges[:-1] + lag_bin_edges[1:]) * 0.5

        # find the bin region
        start_bin = np.argmin(np.abs(lag_bin_edges - start_time))
        stop_bin = np.argmin(np.abs(lag_bin_edges - stop_time))
        lag_bin_edges = lag_bin_edges[start_bin:stop_bin+1] # bins
        lags = lags[start_bin+1:stop_bin+1] # center of the bins
        division_factor = division_factor[start_bin+1:stop_bin+1] # normalization factor

        return lags, division_factor

    def get_intensity_correlations(self, time_bounds, lag_precision):
        time_start = timing.time()
        self.get_file_photons()
        self.time_bounds = time_bounds
        self.lag_precision = lag_precision

        """ Get division factor and bins outside of photon_corr function to store in dot object """
        start_time, stop_time = time_bounds
        cascade_end = int(np.log2(stop_time)) # cascades are collections of lags  with equal bin spacing 2^cascade
        nper_cascade = lag_precision # number of equal
        a = np.array([2**i for i in range(1, cascade_end+1)])
        b = np.ones(nper_cascade)
        division_factor = np.kron(a, b)
        lag_bin_edges = np.cumsum(division_factor/2)
        lags = (lag_bin_edges[:-1] + lag_bin_edges[1:]) * 0.5

        # find the bin region
        start_bin = np.argmin(np.abs(lag_bin_edges - start_time))
        stop_bin = np.argmin(np.abs(lag_bin_edges - stop_time))
        lag_bin_edges = lag_bin_edges[start_bin:stop_bin+1] # bins
        lags = lags[start_bin+1:stop_bin+1] # center of the bins
        division_factor = division_factor[start_bin+1:stop_bin+1] # normalization factor

        self.division_factor = division_factor
        self.lags = lags
        self.lag_bin_edges = lag_bin_edges

        if len(self.file_photons) == len(self.file_stream):
            self.get_sum_signal_all()
        self.get_file_photons()

        for f in self.file_photons:
            if 'sum' not in f: # looking to get cross correlation
                if self.cross_correlations is None:
                    self.cross_correlations = [] # Create empty list where cross correlations are appended
                if self.photons[f].cross_corr is None: # If photons for this stage position are empty, get correlations and append to list. Otherwise, doesn't append to list
                    print(f)
                    print('==============================')
                    self.photons[f].photon_corr(f, 'cross', [0, 1], time_bounds, lag_precision, 0)
                    self.cross_correlations.append(self.photons[f].cross_corr['corr_norm'])
                if self.tau is None:
                    self.tau = self.photons[f].cross_corr['lags']
                    self.length_tau = len(self.tau)

            else: # looking to get auto-correlation for sum signals
                if self.auto_correlations is None:
                    self.auto_correlations = []
                if self.photons[f[11:]].auto_corr is None:
                    print(f)
                    print('==============================')
                    self.photons[f[11:]].photon_corr(f, 'auto', [0,0], time_bounds, lag_precision, 0)
                    self.auto_correlations.append(self.photons[f[11:]].auto_corr['corr_norm'])
                if self.tau is None:
                    self.tau = self.photons[f[11:]].auto_corr['lags']
                    self.length_tau = len(self.tau)

        xcorr = np.array(self.cross_correlations)
        acorr = np.array(self.auto_correlations)


        # Subtract auto-correlation of sum signal from the cross correlation.
        PCFS_interferogram = xcorr - acorr + 1
        PCFS_interferogram = np.transpose(PCFS_interferogram) # Transpose so that main index is tau
        self.PCFS_interferogram = PCFS_interferogram.copy()
        self.g2 = PCFS_interferogram.copy().T
        self.g2_x = xcorr
        self.g2_a = acorr

        time_end = timing.time()
        print('Total time elapsed is %4f s' % (time_end - time_start))


    def g2_gif(self, width=3, height=3, dpi=150, xlim=[1e7, 1e12], ylim=[0.6, 1.05], interval=250, ng2=None, save=False, save_name='pcfs_g2.gif'):
        if ng2 is None:
            ng2 = len(self.cross_correlations)

        Figure, axes = make_fig(width=width, height=height, dpi=dpi)
        def AnimationFunction(frame):
            plt.cla()
            plt.semilogx(self.tau, self.cross_correlations[frame], marker='s', markersize=1, lw=0.5, color='b', label='%.1f ps' % self.optical_delay[frame])
            plt.semilogx(self.tau, self.auto_correlations[frame], marker='s', markersize=1, lw=0.5, color='r')
            dress_fig(xlabel='$\\tau$ (μs)', ylabel='$g^{(2)}(\\tau)$', tight=False, xlim=xlim, ylim=ylim, lgd_loc='lower right')

        anim_created = FuncAnimation(Figure, AnimationFunction, frames=ng2, interval=interval, repeat=True, repeat_delay=1000, save_count=ng2)
        if save:
            anim_created.save(save_name)

        return anim_created

    def correct_autocorr(self):
        acorr_mean = sum(self.auto_correlations)/len(self.auto_correlations)
        # for i in range(0, len(self.auto_correlations)):


    def make_subsection_interferogram(self, t1=1e7, t2=1e11, norm=True):
        # Fot fitting full interferogram
        self.ti1 = find_nearest(t1, self.tau)
        self.ti2 = find_nearest(t2, self.tau)
        self.subI = self.interferogram['Blinking corrected interferogram'][self.ti1:self.ti2, :]
        if norm:
            self.subI = normalize(self.subI, axis=1, norm='max')

    def make_subg2(self, t1=1e7, t2=1e11):
        # Fot fitting full interferogram
        self.ti1 = find_nearest(t1, self.tau)
        self.ti2 = find_nearest(t2, self.tau)
        self.subg2 = self.g2[self.ti1:self.ti2, :]


    def plot_subI(self, clim=([0, 1]), figsize=(4, 4)):
        plt.figure(dpi=150, figsize=figsize)
        plt.contourf(self.optical_delay, self.tau[self.ti1:self.ti2], self.subI, 50)
        plt.clim(clim)
        plt.yscale('log')
        plt.colorbar()
        dress_fig(tight=True, xlabel='$\\delta$ (ps)', ylabel='$\\tau$ (ps)')

    '''
    ============================================================================================
    Analysis of data
    '''


    '''
    This function gets blinking corrected PCFS interferogram.
    '''
    def get_blinking_corrected_PCFS(self):
        self.interferogram = {}
        xcorr = np.array(self.cross_correlations)
        acorr = np.array(self.auto_correlations)
        # self.interferogram['Blinking corrected interferogram'] = 1 - self.cross_correlations / self.auto_correlations
        blinking_corrected_PCFS_interferogram = 1 - xcorr / acorr
        # self.interferogram['Blinking corrected interferogram'] = np.transpose(blinking_corrected_PCFS_interferogram)
        self.interferogram['Blinking corrected interferogram'] = np.transpose(blinking_corrected_PCFS_interferogram)

        # class interferogram:
        #     def __init__(self):
        #         self.raw = np.transpose(blinking_corrected_PCFS_interferogram)
        # self.interferogram = interferogram()
    '''
    This function gets and plots spectral diffusion
    '''
    def plot_interferogram(self, tau_span, fig_dims=(4, 4), dpi=150, normalize=True, root=False, spacer=0, units='ps', marker='s', colors=None, fixframe=False):
        if self.get_blinking_corrected_PCFS is None:
            self.get_blinking_corrected_PCFS()

        # plot PCFS interferogram at different tau
        x = 2 * (self.stage_positions - self.white_fringe) # in mm
        x = x/(299792458)/1000*1e12 # convert to ps
        self.optical_delay = x

        if colors is None:
            colors = np.array(sns.color_palette("icefire", 12))
        if fixframe:
            fig = make_fig(figsize=fig_dims, dpi=dpi)
        else:
            fig = plt.figure(dpi=150, figsize=fig_dims)

        # plt.subplot(2, 1, 1)
        for i in range(len(tau_span)):
            t1 = tau_span[i][0]
            t2 = tau_span[i][1]
            if units == 'µs':
                t1 = t1*1e6
                t2 = t2*1e6
            y = self.interferogram['Blinking corrected interferogram'][fnear(t1, self.tau):fnear(t2, self.tau), :].mean(axis=0)
            # plt.scatter(x, y, marker='s', s=1, color=colors[i], label='%1.0e us' % tau_select[i])
            if normalize:
                y = y / max(y)
            if root:
                y = np.sqrt(y)
            plt.plot(x, y+i*spacer, marker=marker, ms=2, color=colors[i], label=(eformat(t1, 0, 0) + '-' + eformat(t2, 0, 0) + ' ' + units))
            # plt.plot(x, y + i * spacer, marker='s', ms=2, color=colors[i], label='%.0f - %.0f µs' % (tau_span[i][0], tau_span[i][1]))

        plt.ylabel(r'1 - $g^{(2)}_{x}/g^{(2)}_{a}$')
        plt.xlabel('$\\delta$ (ps)')
        plt.legend(loc='upper right')

        dress_fig()
        if not fixframe:
            plt.tight_layout()
        plt.show()

        return fig

    def plot_mirror_spectral_corr(self, tau_span, xlim=[-3, 3], fig_dims=(4, 4), spacer=0, normalize=True, subtract_min=False,
                                  plot_fit=False, units='ps', marker='s', colors=None, fixframe=False, dpi=150, ncols=3):
        x = self.mirror_spectral_correlation['ζ']
        if colors is None:
            colors = np.array(sns.color_palette("icefire", 12))

        if fixframe:
            fig = make_fig(fig_dims, dpi=dpi)
        else:
            fig = plt.figure(dpi=dpi, figsize=fig_dims)
        for i in range(len(tau_span)):
            t1 = tau_span[i][0]
            t2 = tau_span[i][1]
            if units == 'µs':
                t1 = t1*1e6
                t2 = t2*1e6
            y = self.mirror_spectral_correlation['spectral_corr'][fnear(t1, self.tau):fnear(t2, self.tau), :].mean(axis=0)
            if subtract_min:
                y = y - min(y)
            if normalize:
                y = y/max(y)

            # if plot_fit:
            #     plt.scatter(x, y + i * spacer, marker=marker, s=2, color=colors[i], label=('%1.0e - %1.0e ' + units) % (tau_span[i][0], tau_span[i][1]))
            # else:
            if plot_fit:
                # Γ = self.interferogram['mle'][i].theta[0] * 1000  # fwhm in µeV
                # label = ('$\\tau_{%1.0d→%1.0d}$' % (2 * i, 2 * i + 1)) + ', %.1f µeV' % Γ
                label = ('$\\tau_{%1.0d→%1.0d}$' % (2 * i, 2 * i + 1))
            else:
                label = ('$\\tau_{%1.0d→%1.0d}$' % (2*i, 2*i+1))

            plt.plot(x, y+i*spacer, marker=marker, ms=2, color=colors[i],
                     # label=('%1.0e - %1.0e ' + units) % (tau_span[i][0], tau_span[i][1]),
                     label=label,
                     )
            # plt.plot(x, y + i * spacer, marker='s', ms=2, color=colors[i],label='%.0f - %.0f µs' % (tau_span[i][0], tau_span[i][1]))

            if plot_fit:
                y = self.interferogram['fitted_spectral_corr'][i]

                if normalize:
                    y = y/max(y)
                plt.plot(x, y+i*spacer, color=colors[i]*0.3)

        dress_fig(xlabel=r'$\zeta$ (meV)', ylabel=r'$p(\zeta)$', xlim=xlim, legend=True, lgnd_cols=ncols)


    def get_mirror_spectral_corr(self, white_fringe_pos, upsample=1):
        end = -1
        # construct mirrored data
        interferogram = self.interferogram['Blinking corrected interferogram'][:, :end]

        white_fringe_ind = find_nearest(self.stage_positions, white_fringe_pos)

        x = 2*(self.stage_positions - white_fringe_pos) # in mm
        x = x/(299792458)/1000*1e12 # convert to ps
        self.optical_delay = x
        self.white_fringe = white_fringe_pos
        self.wf = white_fringe_ind
        mirror_intf = np.hstack((np.fliplr(interferogram[:, white_fringe_ind:]), interferogram[:, white_fringe_ind+1:]))
        temp = white_fringe_pos - self.stage_positions[white_fringe_ind:end]
        temp = temp[::-1]
        mirror_stage_pos = np.hstack((temp, self.stage_positions[white_fringe_ind+1:end] - white_fringe_pos))
        # interp_stage_pos = np.arange(min(mirror_stage_pos), max(mirror_stage_pos), 0.1)

        interp_N = upsample*round((max(mirror_stage_pos) - min(mirror_stage_pos))/0.1)

        # Force odd number of interpolated points
        if (interp_N % 2) == 0:
            interp_N += 1

        interp_stage_pos = np.linspace(start=min(mirror_stage_pos), stop=max(mirror_stage_pos), num=interp_N)

        # row-wise interpolation
        a, b = mirror_intf.shape
        interp_mirror = np.zeros((a, len(interp_stage_pos)))
        for i in range(a):
            interp_mirror[i, :] = np.interp(interp_stage_pos, mirror_stage_pos, mirror_intf[i, :])

        self.interferogram['mirror_stage_positions'] = mirror_stage_pos
        self.interferogram['mirror_interferogram']   = interp_mirror
        self.interferogram['interp_stage_pos']       = interp_stage_pos

        self.interp_stage_pos = interp_stage_pos
        optical_delay =  2*interp_stage_pos/(3e8)/1000*1e12

        # some constants
        eV2cm = 8065.54429
        cm2eV = 1 / eV2cm
        N = len(interp_stage_pos)
        path_length_difference = 0.2 * (interp_stage_pos) # NOTE: This is where we convert to path length difference space in cm.
        delta = (max(path_length_difference) - min(path_length_difference)) / (N-1)
        ζ_eV = np.fft.fftshift(np.fft.fftfreq(N, delta)) * cm2eV * 1000 # in meV

        # Use these later for fitting interferogram
        self.Fourier = {
            'path_length_difference': path_length_difference,
            'delta': delta,
            'interp_stage_pos': interp_stage_pos,
            'optical_delay': optical_delay,
        }

        # get reciprocal space (wavenumbers).
        # increment = 1 / delta
        # ζ_eV = np.linspace(-0.5 * increment, 0.5 * increment, num = N) * cm2eV * 1000 # converted to meV

        # take the FFT of the interferogram to get the spectral correlation. All that shifting is to shift the zero frequency component to the middle of the FFT vector. We take the real part of the FFT because the interferogram is by definition entirely symmetric.

        spectral_correlation = self.interferogram['mirror_interferogram'].copy()
        for i in range(a):
            spectral_correlation[i, :] = np.abs(np.fft.fftshift(np.fft.fft(self.interferogram['mirror_interferogram'][i,:])))

        self.mirror_spectral_correlation = {}
        self.mirror_spectral_correlation['spectral_corr'] = spectral_correlation
        self.mirror_spectral_correlation['ζ'] = ζ_eV
        self.spectrum_energy = self.energy_diff_to_energy(ζ_eV)


    def get_splev_mirror_spec_corr(self, white_fringe_pos, white_fringe_ind,stage_increment=0.005):
        '''
        Using spline interpolation
        '''
        # construct mirrored data
        interferogram = self.interferogram['Blinking corrected interferogram'][:,:]
        mirror_intf = np.hstack((np.fliplr(interferogram[:, white_fringe_ind:]), interferogram[:, white_fringe_ind+1:]))
        temp = white_fringe_pos - self.stage_positions[white_fringe_ind:]
        temp = temp[::-1]
        mirror_stage_pos = np.hstack((temp, self.stage_positions[white_fringe_ind+1:] - white_fringe_pos))
        interp_stage_pos = np.arange(min(mirror_stage_pos), max(mirror_stage_pos)+stage_increment, stage_increment )

        # row-wise interpolation
        a,b = mirror_intf.shape
        interp_mirror = np.zeros((a,len(interp_stage_pos)))
        for i in range(a):
            x = mirror_stage_pos
            y = mirror_intf[i,:]
            tck = interpolate.splrep(x, y, s=0)
            xnew = interp_stage_pos
            ynew = interpolate.splev(xnew, tck, der=0)
            interp_mirror[i,:] = ynew

        self.mirror_stage_positions = mirror_stage_pos
        self.interferogram['mirror_interferogram'] = interp_mirror # not including the first line of position

        #some constants
        eV2cm = 8065.54429
        cm2eV = 1 / eV2cm

        N = len(interp_stage_pos)
        path_length_difference = 0.2 * (interp_stage_pos) # NOTE: This is where we convert to path length difference space in cm.
        delta = (max(path_length_difference) - min(path_length_difference)) / (N-1)
        ζ_eV = np.fft.fftshift(np.fft.fftfreq(N, delta)) * cm2eV * 1000 # in meV

        # get reciprocal space (wavenumbers).
        # increment = 1 / delta
        # ζ_eV = np.linspace(-0.5 * increment, 0.5 * increment, num = N) * cm2eV * 1000 # converted to meV

        # take the FFT of the interferogram to get the spectral correlation. All that shifting is to shift the zero frequency component to the middle of the FFT vector. We take the real part of the FFT because the interferogram is by definition entirely symmetric.
        spectral_correlation = self.interferogram['mirror_interferogram'].copy()
        for i in range(a):
            spectral_correlation[i,:] = np.abs(np.fft.fftshift(np.fft.fft(self.interferogram['mirror_interferogram'][i,:])))

        self.splev_spec_corr = {}
        self.splev_spec_corr['spectral_corr'] = spectral_correlation
        self.splev_spec_corr['ζ'] = ζ_eV

    def plot_splev_spec_corr(self, tau_select, xlim):
        x = self.splev_spec_corr['ζ']
        ind = np.array([np.argmin(np.abs(self.tau - tau)) for tau in tau_select])
        legends = [tau/1e9 for tau in tau_select]
        y = self.splev_spec_corr['spectral_corr'][ind,:]

        plt.figure()
        plt.subplot(2,1,1)
        for i in range(len(tau_select)):
            plt.plot(x, y[i,:])

        plt.ylabel(r'$p(\ζ)$')
        plt.xlabel(r'$\ζ$ [meV]')
        plt.xlim(xlim)
        plt.legend(legends)
        plt.title(self.PCFS_ID + r' Mirrored Spectral Correlation at $\tau$ [ms]')

        plt.subplot(2,1,2)
        for i in range(len(tau_select)):
            plt.plot(x, y[i,:]/max(y[i,:]))

        plt.ylabel(r'Normalized $p(\ζ)$')
        plt.xlabel(r'$\ζ$ [meV]')
        plt.xlim(xlim)
        plt.legend(legends)

        plt.title(self.PCFS_ID + r' Mirrored Spectral Correlation at $\tau$ [ms]')
        plt.tight_layout()
        plt.show()

    def plot_spectral_corr(self, stage_positions, interferogram, white_fringe_pos):

        #some constants
        eV2cm = 8065.54429
        cm2eV = 1 / eV2cm

        N = len(stage_positions)
        path_length_difference = 2 * (stage_positions - white_fringe_pos) * 0.1 # NOTE: This is where we convert to path length difference space in cm.
        delta = (max(path_length_difference) - min(path_length_difference)) / (N-1)
        ζ_eV = np.fft.fftshift(np.fft.fftfreq(N, delta)) * cm2eV * 1000 # in meV

        # get reciprocal space (wavenumbers).
        # increment = 1 / delta
        # ζ_eV = np.linspace(-0.5 * increment, 0.5 * increment, num = N) * cm2eV * 1000 # converted to meV

        # take the FFT of the interferogram to get the spectral correlation. All that shifting is to shift the zero frequency component to the middle of the FFT vector. We take the real part of the FFT because the interferogram is by definition entirely symmetric.
        spectral_correlation = np.abs(np.fft.fftshift(np.fft.fft(interferogram)))

        normalized_spectral_correlation = spectral_correlation / max(spectral_correlation)


        plt.plot(ζ_eV, normalized_spectral_correlation, '-o', markersize = 1)

        plt.ylabel(r'Normalized $p(\zeta)$')
        plt.xlabel(r'$\zeta$ (meV)')

        plt.title(self.PCFS_ID + r' Spectral Correlation at $\tau$ [ms]')
        dress_fig()
        plt.show()

    def fit_spectral_corr(self, funk, params, tau_span, units='ps', nruns=1, plotting=True, fig_dims=(4, 4)):
        mles = []
        ζ = self.mirror_spectral_correlation['ζ']

        for i in range(len(tau_span)):
            t1 = tau_span[i][0]
            t2 = tau_span[i][1]
            if units == 'µs':
                t1 = t1*1e6
                t2 = t2*1e6

            y = self.mirror_spectral_correlation['spectral_corr'][fnear(t1 , self.tau):fnear(t2, self.tau), :].mean(axis=0)
            y = y/max(y)
            # y = y-min(y)

            theta = params[0]
            var_names = params[1]

            e = ζ_to_energy(ζ)
            # Create a wider and finer energy array for generating spectrum, then generate corresponding ζ and δ axes
            upsampled_e = np.linspace(5*e[0], 5*e[-1], 10*len(e))
            upsampled_ζ = energy_to_ζ(upsampled_e)

            print("Maximum Likelihood Estimation of spectral correlation...")
            mle = MLE(
                theta_in=theta,
                args=(upsampled_e, upsampled_ζ, ζ, y, funk),
                nruns=nruns,
                obj_function=loss_AutoCorr_MSE,
                randomizer='bounded',
                bounds=([theta['lb'], theta['ub']]),
                var_names=var_names
            )
            mles.append(mle)

            with np.printoptions(precision=5, suppress=True):
                print(mle.theta)

            spectrum = make_spectrum(e, funk, mle.theta)
            ycorr = correlate(spectrum, spectrum)
            ycorr = ycorr / max(ycorr) + mle.theta[-1]
            ycorr = ycorr / max(ycorr)

            self.mirror_spectral_correlation['mle'] = mles
            self.mirror_spectral_correlation['spectrum'] = spectrum

            if plotting:
                plt.figure(dpi=150, figsize=fig_dims)
                plt.plot(ζ, y, label='Raw')
                plt.plot(ζ, ycorr, label='Fit')
                plt.legend()
                dress_fig()


    def plot_correlations(self, delays, fig_dims=(4, 4), dpi=150, xlim=[1e5, 1e12], ylim=[0.5, 1.5], units='ps',
                           autocorr=False, crosscorr=True, g2=False, ML=False, ML_var=False, nstd=1, makefig=False, colors=None, fixframe=False, legend_on=True,
                           ):
        self.xcoi = [] # xcoi = cross correlations of interest
        self.acoi = [] # acoi = auto correlations of interest
        self.g2oi = [] # g2oi = g2 of interest
        if fixframe:
            make_fig(fig_dims, dpi=150)
        else:
            plt.figure(dpi=dpi, figsize=fig_dims)

        if colors is None:
            colors = np.array(sns.color_palette("icefire", 12))

        tau = self.tau.copy()
        if units == 'µs':
            tau = tau/1e6
            xlim = xlim/1e6

        for ctr, i in enumerate(delays):
            i = find_nearest(self.optical_delay, i)
            if autocorr:
                plt.plot(tau, self.auto_correlations[i], color=colors[ctr], label='')
                self.acoi.append(self.auto_correlations[i])
            if crosscorr:
                plt.plot(tau, self.cross_correlations[i], color=colors[ctr], label='δ = %.2f ps' % self.optical_delay[i], zorder=0)
                self.xcoi.append(self.cross_correlations[i])
            if g2:
                plt.plot(tau, self.g2[i, :], color=colors[ctr], label='δ = %.2f ps' % self.optical_delay[i])
                self.g2oi.append(self.g2[i, :])
            if ML:
                τ = self.tau_interp
                μ = self.g2s_pred_μ[i, :]
                σ = self.g2s_pred_σ[i, :]
                plt.plot(τ, μ, color=colors[ctr]*0.2, zorder=10)
                if ML_var:
                    fill_col = [0, 0, 0]
                    plt.fill_between(τ, μ+nstd*σ, μ-nstd*σ, linewidth=0.0, color=fill_col, alpha=0.2, zorder=5
                                     # label='$\mu$ +/- %d$\sigma$' % nstd
                                     )

        plt.xscale('log')
        dress_fig(
            xlabel=r'$\tau$ ('+units+')',
            ylabel=r'$g^{(2)}(τ)$',
            xlim=xlim,
            ylim=ylim,
            tight=not fixframe,
            legend=legend_on,
            lgnd_cols=3
        )

    @staticmethod
    def energy_diff_to_energy(x):
        n = (len(x) - 1) / 2
        delta = x[1] - x[0]  # energy difference
        return np.linspace(-n/2*delta, n/2*delta, ceil(n)+1)

    def generate_interferogram(self, funk, theta):
        spectrum    = function_clean(theta, self.spectrum_energy, funk)
        spectrum_ac = autocorrelate_peaks(x=self.mirror_spectral_correlation['ζ'], theta=theta, function=funk)
        spectrum_ac_fft = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(spectrum_ac))))
        return spectrum, spectrum_ac, spectrum_ac_fft

    def fit_interferogram(self, funk, params, tau_span, rooted=True, units='ps', nruns=1, randomizer=None):
        ys = []
        mles = []
        upsampled_spectra = []
        spec_corrs = []
        Is = []
        decay_constants = []
        coherence_decays = []
        t1s = []
        t2s = []
        idxs = []
        for i in range(len(tau_span)):
            t1 = tau_span[i][0]
            t2 = tau_span[i][1]
            t1s.append(t1)
            t2s.append(t2)
            if units == 'µs':
                t1 = t1*1e6
                t2 = t2*1e6

            ζ = self.mirror_spectral_correlation['ζ']
            y = self.interferogram['mirror_interferogram'][fnear(t1, self.tau):fnear(t2, self.tau), :].mean(axis=0)
            y = y/max(y); ys.append(y)
            δ = 2*self.interp_stage_pos/(299792458)/1000*1e12
            theta = params[0]
            var_names = params[1]

            e = ζ_to_energy(ζ)
            # Create a wider and finer energy array for generating spectrum, then generate corresponding ζ and δ axes
            upsampled_e = np.linspace(5*e[0], 5*e[-1], 10*len(e))
            upsampled_ζ = energy_to_ζ(upsampled_e)
            upsampled_δ = fourier_ζ_to_ps(upsampled_ζ)

            print("Maximum Likelihood Estimation of interferogram...")
            mle = MLE(
                theta_in=theta,
                args=(upsampled_e, upsampled_δ, δ, y, funk),
                nruns=nruns,
                obj_function=loss_AutoCorrFFT,
                # obj_function=loss_interferogram_wiener,
                guess='random',
                randomizer=randomizer,
                var_names=var_names,
            )
            mles.append(mle)

            # def spectrum_to_I_interp(theta, function, upsampled_e, upsampled_δ, δ, norm=True):
            upsampled_spectrum  = make_spectrum(upsampled_e, funk, mle.theta)
            upsampled_spectra.append(upsampled_spectrum)
            upsampled_spec_corr = correlate(upsampled_spectrum, upsampled_spectrum)
            upsampled_I         = spec_corr_to_interferogram_FFT(upsampled_spec_corr)

            spec_corr = np.interp(ζ, upsampled_ζ, upsampled_spec_corr)
            spec_corr = spec_corr/max(spec_corr)
            spec_corrs.append(spec_corr)

            I = np.interp(δ, upsampled_δ, upsampled_I)
            I = I/max(I)
            Is.append(I)

            self.spectrum_energy = upsampled_e
            idx   = fnear(max(y), y)
            idxs.append(idx)

            from conversions import eV_to_wn, meV_to_T2
            # Coherence time
            gamma1 = mle.theta[0]
            # decay_constant = (1/eV_to_angfreq(gamma1/1000))*1e12
            # decay_constant = eV_to_wn(gamma1*2*3.1459/10)
            decay_constant = meV_to_T2(gamma1)
            decay_constants.append(decay_constant)
            amplitude = I[idx+1]
            coherence_decay = amplitude * np.exp(-self.Fourier['optical_delay'][idx:]*(2/decay_constant))
            coherence_decays.append(coherence_decay)

        self.interferogram['fit_function'] = funk
        self.interferogram['data_for_fit'] = ys
        self.interferogram['mle'] = mles
        self.interferogram['fitted_spectrum'] = upsampled_spectra
        self.interferogram['fitted_spectral_corr'] = spec_corrs
        self.interferogram['fitted_interferogram'] = Is
        self.interferogram['decay_constant'] = decay_constants
        self.interferogram['coherence_decay'] = coherence_decays
        self.interferogram['units'] = units
        self.interferogram['t1'] = t1s
        self.interferogram['t2'] = t2s
        self.Fourier['max_idx'] = idxs

    def plot_interferogram_fit(self, fig_dims=(4, 4), coherence_amp=1, marker='s', fixframe=False, dpi=150, ncol=1, spacer=0, colors=None):
        if colors is None:
            colors = np.array(sns.color_palette("icefire", 12))
        if fixframe:
            fig = make_fig(fig_dims, dpi)
        else:
            fig = plt.figure(dpi=150, figsize=fig_dims)

        for i in range(0, len(self.interferogram['data_for_fit'])):
            idx   = self.Fourier['max_idx'][i]
            xplot = self.Fourier['optical_delay']
            y     = self.interferogram['data_for_fit'][i]
            units = self.interferogram['units']
            t1    = self.interferogram['t1'][i]
            t2    = self.interferogram['t2'][i]
            I     = self.interferogram['fitted_interferogram'][i]

            plt.plot(xplot[idx:], y[idx:]+i*spacer,  marker=marker, ms=2, color=colors[i],
                     # label='%.0f - %.0f µs' % (tspan[0], tspan[1]),
                     # label=('%1.0e-%1.0e' + units) % (t1, t2),
                     # label=(eformat(t1, 0, 0) + '-' + eformat(t2, 0, 0) + ' ' + units),
                     label=('$\\tau_{%1.0d→%1.0d}$' % (2 * i, 2 * i + 1)),
                     )


            plt.plot(xplot[idx:], I[idx:]+i*spacer, color=colors[0]*0.2)
            # plt.plot(xplot[idx:], exp_coherence_decay, '--', label='Coherence', color=[0.8, 0.2, 0.4])
            plt.plot(xplot[idx:], (self.interferogram['coherence_decay'][i]*coherence_amp)+i*spacer, '--', label='$T_2$ = %.0f ps' % self.interferogram['decay_constant'][i], color=[0.8, 0.2, 0.4])
            # if disp_decay_const:
            #     plt.text(x=xplot[-40], y=amplitude, s='$T_2$ = %.1f ps' % decay_constant, color=[0.8, 0.2, 0.4])

            plt.legend()
            if fixframe:
                tight = False
            else:
                tight = True
            dress_fig(tight=tight, ylabel=(r'1 - $g^{(2)}_{x}/g^{(2)}_{a}$'), xlabel='$\\delta$ (ps)', lgnd_cols=ncol)

            with np.printoptions(precision=5, suppress=True):
                print(self.interferogram['mle'][i].theta)

        return fig

    def plot_spectrum(self, fig_dims=(4, 4), xlim=[-3, 3], ylim=None, resolution=1, norm=True, subtract_min=False, baseline=1e-5,
                      fixframe=False, dpi=150, ncol=1, spacer=0, colors=None):
        if colors is None:
            colors = np.array(sns.color_palette("icefire", 12))
        if fixframe:
            fig = make_fig(fig_dims, dpi)
        else:
            fig = plt.figure(dpi=150, figsize=fig_dims)

        if resolution > 1:
            energy_ax = np.linspace(self.spectrum_energy[0], self.spectrum_energy[-1], len(self.spectrum_energy)*int(resolution))
        else:
            energy_ax = self.spectrum_energy

        for i, mle in enumerate(self.interferogram['mle']):
            t1    = self.interferogram['t1'][i]
            t2    = self.interferogram['t2'][i]
            Γ = self.interferogram['mle'][i].theta[0] * 1000  # fwhm in µeV
            spectrum = function_clean(mle.theta, energy_ax, self.interferogram['fit_function'])
            if subtract_min:
                spectrum = spectrum-min(spectrum)+baseline
            if norm:
                spectrum = spectrum/max(spectrum)

            plt.plot(
                energy_ax, spectrum+i*spacer, color=colors[i],
                label=('$\\tau_{%1.0d→%1.0d}$' % (2*i, 2*i+1)) + ', %.1f µeV' % Γ,
                # label=('$\\tau_{%1.0d→%1.0d}$' % (2 * i, 2 * i + 1)),
            )
            if fixframe:
                tight=False
            else:
                tight=True
            dress_fig(tight=tight, xlim=xlim, ylim=ylim, xlabel='Energy (meV)', ylabel='Amplitude', lgnd_cols=ncol)

    #     # try:
    #     spectrum = function_clean(mle.theta, self.spectrum_energy, funk)
    #     self.interferogram['mle']
    #     x = self.spectrum_energy
        # except:
        #     print('Get interferogram MLE before plotting fitted spectrum')
        #     pass

    '''
    This function gets the fourier spectrum from the photon stream.
    '''
    def get_Fourier_spectrum_from_stream(self, bin_width, file_in):
        t = np.zeros(len(self.stage_positions)) # for intensity
        Fourier = np.zeros(len(self.stage_positions))
        self.get_file_photons()

        for f in self.file_photons:
            # looking to get cross correlation
            if 'sum' not in f:
                if file_in in f:
                    correlation_number = int(re.findall(r'\d+', f)[0]) # extract the number of correlation measurements from the file names
                    self.photons[f].get_intensity_trace(f, bin_width)
                    intensity = self.photons[f].intensity_counts['Trace']
                    t[correlation_number] = (np.sum(intensity[:,0]) + np.sum(intensity[:,1]))
                    Fourier[correlation_number] = (np.sum(intensity[:,0]) - np.sum(intensity[:,1])) / t[correlation_number]

        out_dic = {}
        out_dic['Fourier'] = Fourier
        out_dic['stage_positions'] = self.stage_positions
        out_dic['intensity'] = t
        self.Fourier[file_in] = out_dic


def spec_corr_to_interferogram_FFT(spec_corr):
    return np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(spec_corr))))
    # return np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(spec_corr))))/len(spec_corr)
    # return np.fft.ifft(np.fft.ifftshift(spec_corr))

def interferogram_to_spec_corr_FFT(interferogram):
    return np.abs(np.fft.fftshift(np.fft.fft(interferogram)))
    # return np.fft.fftshift(np.fft.fft(interferogram))

def spectrum_to_I_interp(theta, function, upsampled_e, upsampled_δ, δ, norm=True):
    spectrum = make_spectrum(upsampled_e, function, theta)
    spec_corr = correlate(spectrum, spectrum)
    upsampled_I = spec_corr_to_interferogram_FFT(spec_corr)
    I = np.interp(δ, upsampled_δ, upsampled_I)
    if norm:
        I = I/max(I)
    return I

""" Spectrum functions for fitting spectral correlations """
def make_spectrum(x, function, theta):
    w = np.array(theta)
    f = eval(function)
    return f


""" FITTING FUNCTIONS AND PARAMETER ARRAYS """

# One Gaussian with baseline
SingleGaussian = "w[0] * np.exp(-(x)**2/(2*w[1]**2)) + 0*w[2]"

                                         # a1   s1    y0
theta_SingleGaussian = {'init': np.array([0.2,  20, 0.01]),
                          'lb': np.array([0.0,  10, 0.00]),
                          'ub': np.array([1.0, 100, 0.05])
                       }

# Two Gaussians with variable offset and baseline
DoubleGaussian = "w[0] * np.exp(-(x)**2/(2*w[1]**2)) + w[2] * np.exp(-(x-w[4])**2/(2*w[3]**2)) + 0*w[5]"

                                         # a1   s1   a2   s2    a    y0
theta_DoubleGaussian = {'init': np.array([0.3,  21, 0.2,  40,   5, 0.01]),
                          'lb': np.array([0.2,  10, 0.0,  10,   0, 0.00]),
                          'ub': np.array([1.0, 100, 0.1, 100,  10, 0.05])
                       }

# Double Gaussian with no offset
DoubleGaussian_noOffset = "w[0] * np.exp(-(x)**2/(2*w[1]**2)) + w[2] * np.exp(-(x)**2/(2*w[3]**2)) + 0*w[4]"

                                                  # a1   s1   a2   s2   y0
theta_DoubleGaussian_noOffset = {'init': np.array([0.4,  20, 0.1,  60, 0.01]),
                                   'lb': np.array([0.1,  10, 0.0,  10, 0.00]),
                                   'ub': np.array([1.0, 100, 0.1, 100, 0.05])
                                }

# One Lorentzian with baseline
SingleLorentzian = "   w[0] * (0.5*w[1]) / ( x**2 + (0.5*w[1])**2 ) " \
                   " + 0*w[2] "


                                           # a1   g1    y0
theta_SingleLorentzian = {'init': np.array([0.2, 1e-1, 0.01]),
                            'lb': np.array([0.0, 1e-3, 0.00]),
                            'ub': np.array([1.0, 1e+0, 0.05])
                         }

# One Lorentzian
Lorentzian       = "   w[0]/np.pi * (0.5*w[1]) / ( (x-w[2])**2 + (0.5*w[1])**2 ) + w[3]"

                                           # a1    g1    o    y0
theta_Lorentzian       = {'init': np.array([0.2, 1e-1, 0.0, 0.01]),
                            'lb': np.array([0.0, 1e-3, 0.0, 0.00]),
                            'ub': np.array([1.0, 1e+0, 5.0, 0.05])
                         }

# Two Lorentzians with variable offset and baseline
DoubleLorentzian = "   w[0] * (0.5*w[1]) / ( (x     )**2 + (0.5*w[1])**2 ) " \
                   " + w[2] * (0.5*w[3]) / ( (x-w[4])**2 + (0.5*w[3])**2 ) " \
                   " + 0*w[5] "

                                           # a1    g1   a2    g2     o    y0
theta_DoubleLorentzian = {'init': np.array([0.1, 1e-2, 0.1, 1e-2,  0.1, 0.01]),
                            'lb': np.array([0.0, 1e-4, 0.0, 1e-4,  0.0, 0.00]),
                            'ub': np.array([1.0, 2e-1, 1.0, 2e-1,  0.5, 0.05])
                         }


# Two Lorentzians with common gamma, variable offset and baseline
twoLorentzian_oneFWHM = "   w[1] * (0.5*w[0]) / ( (x     )**2 + (0.5*w[0])**2 ) " \
                        " + w[2] * (0.5*w[0]) / ( (x-w[3])**2 + (0.5*w[0])**2 ) " \
                        " + 0*w[4] "

                                                       # a1    g1   a2    o    y0
theta_DoubleLorentzian_SingleGamma = {'init': np.array([0.2, 1e-2, 0.2, 0.1, 0.01]),
                                        'lb': np.array([0.0, 1e-4, 0.0, 0.0, 0.00]),
                                        'ub': np.array([0.1, 2e-1, 0.1, 0.5, 0.05])
                                     }

# Two Lorentzians with common gamma, variable offset and baseline
TwoLorentz_OneGamma_offset   = "   w[0] * (0.5*w[2]) / ( (x-w[4]     )**2 + (0.5*w[2])**2 ) " \
                               " + w[1] * (0.5*w[2]) / ( (x-w[4]-w[3])**2 + (0.5*w[2])**2 ) " \
                               " + w[5] "

theta_TwoLorentz_OneGamma_offset = {
                     # a1,   a2,   Γ,  o1,   x0,   y0
    'init': np.array([0.2, 1e-2, 0.2, 0.1, 2400, 0.01]),
      'lb': np.array([0.0, 1e-4, 0.0, 0.0, 2200, 0.00]),
      'ub': np.array([0.1, 2e-1, 0.1, 0.5, 2600,  0.05])
}


oneLorentz = "   w[1] * (0.5*w[0]) / ( (x)**2 + (0.5*w[0])**2 ) + w[2]"

def make_theta_oneLorentzian(params_filepath):
    p = pd.read_csv(params_filepath, index_col=0)
    var_names = ['Γ',  'a1', 'y0']
    return {                     #   Γ,           a1,              y0
        'init': np.array([p.iloc[0][0], p.iloc[0][5], p.iloc[0][11]]),
          'lb': np.array([p.iloc[1][0], p.iloc[1][5], p.iloc[1][11]]),
          'ub': np.array([p.iloc[2][0], p.iloc[2][5], p.iloc[2][11]]),
    }, var_names


def make_theta_twoLorentzian_oneFWHM(params_filepath):
    p = pd.read_csv(params_filepath, index_col=0)
    var_names = ['Γ',  'a1', 'a2', 'energy offset', 'y0']
    return {                     #   Γ,           a1,           a2,            o,           y0
        'init': np.array([p.iloc[0][0], p.iloc[0][5], p.iloc[0][6], p.iloc[0][8], p.iloc[0][11]]),
          'lb': np.array([p.iloc[1][0], p.iloc[1][5], p.iloc[1][6], p.iloc[1][8], p.iloc[1][11]]),
          'ub': np.array([p.iloc[2][0], p.iloc[2][5], p.iloc[2][6], p.iloc[2][8], p.iloc[2][11]]),
    }, var_names


# Three Lorentzians with single gamma
threeLorentzian_oneFWHM = "   w[1] * (0.5*w[0]) / ( (x)**2 + (0.5*w[0])**2 ) " \
                          " + w[2] * (0.5*w[0]) / ( (x-w[4])**2 + (0.5*w[0])**2 ) " \
                          " + w[3] * (0.5*w[0]) / ( (x-w[5])**2 + (0.5*w[0])**2 ) " \
                          " + w[6] "


def make_theta_threeLorentzian_oneFWHM(params_filepath):
    if params_filepath.split('.')[-1] == 'xls':
        p = pd.read_excel(params_filepath, index_col=0)
    elif params_filepath.split('.')[-1] == 'csv':
        p = pd.read_csv(params_filepath, index_col=0)
    var_names = ['Γ',  'a1', 'a2', 'a3', 'energy offset1', 'energy offset2', 'y0']
    return {                     #   Γ,           a1,           a2,           a3,           o1,           o1,           y0
        'init': np.array([p.iloc[0][0], p.iloc[0][4], p.iloc[0][5], p.iloc[0][6], p.iloc[0][8], p.iloc[0][9], p.iloc[0][11]]),
          'lb': np.array([p.iloc[1][0], p.iloc[1][4], p.iloc[1][5], p.iloc[1][6], p.iloc[1][8], p.iloc[1][9], p.iloc[1][11]]),
          'ub': np.array([p.iloc[2][0], p.iloc[2][4], p.iloc[2][5], p.iloc[2][6], p.iloc[2][8], p.iloc[2][9], p.iloc[2][11]]),
    }, var_names


# # Three Lorentzians with variable offset and baseline
# TripleLorentzian = "   w[0] * (0.5*w[1]) / ( (x     )**2 + (0.5*w[1])**2 ) " \
#                    " + w[2] * (0.5*w[3]) / ( (x-w[6])**2 + (0.5*w[3])**2 ) " \
#                    " + w[4] * (0.5*w[5]) / ( (x-w[7])**2 + (0.5*w[5])**2 ) " \
#                    " + 0*w[8] "

# Three Lorentzians with variable offset and baseline
TripleLorentzian = "   w[0] * (0.5*w[1]) / ( (x-w[6]     )**2 + (0.5*w[1])**2 ) " \
                   " + w[2] * (0.5*w[3]) / ( (x-w[6]-w[7])**2 + (0.5*w[3])**2 ) " \
                   " + w[4] * (0.5*w[5]) / ( (x-w[6]-w[8])**2 + (0.5*w[5])**2 ) " \
                   " + 0*w[9] "


                                           # a1    g1   a2    g2   a3    g3   og   o1   o1    y0
theta_TripleLorentzian = {'init': np.array([0.1, 1e-2, 0.1, 1e-2, 0.1, 1e-2, 0.1, 0.1, 0.0, 0.01]),
                            'lb': np.array([0.0, 1e-4, 0.0, 1e-4, 0.0, 1e-4, 0.0, 0.0, 0.0, 0.00]),
                            'ub': np.array([1.0, 1e+0, 1.0, 1e+0, 1.0, 1e+0, 2.0, 2.0, 0.0, 0.05])
                         }


# # Three Lorentzians with variable offset and baseline
# TripleLorentzian_SingleGamma = "   w[0] * (0.5*w[3]) / ( (x-w[4])**2 + (0.5*w[3])**2 ) " \
#                                " + w[1] * (0.5*w[3]) / ( (x-w[4]-w[5])**2 + (0.5*w[3])**2 ) " \
#                                " + w[2] * (0.5*w[3]) / ( (x-w[4]-w[6])**2 + (0.5*w[3])**2 ) " \
#                                " + 0*w[7] "

                                                       # a1   a2   a3    g1   og    o1   o2    y0
theta_TripleLorentzian_SingleGamma = {'init': np.array([0.1, 0.1, 0.1, 1e-1, 0.0,  0.5, 1.5, 0.01]),
                                        'lb': np.array([0.0, 0.0, 0.0, 1e-4, 0.0,  0.0, 0.0, 0.00]),
                                        'ub': np.array([1.0, 1.0, 1.0, 1e-1, 10.0, 1.0, 2.0, 0.10])
                                     }



# Three Lorentzians with variable offset, a common Lorentzian FWHM, one broad Gaussian, and baseline
TripleLorentzian_OneGaussian_SingleGamma = "   w[0] * (0.5*w[1]) / ( (x     )**2 + (0.5*w[1])**2 ) " \
                                           " + w[2] * (0.5*w[1]) / ( (x-w[4])**2 + (0.5*w[1])**2 ) " \
                                           " + w[3] * (0.5*w[1]) / ( (x-w[5])**2 + (0.5*w[1])**2 ) " \
                                           " + w[6] * np.exp(-(x-w[8])**2/(2*w[7]**2)) " \
                                           " + 0*w[9] "

# Three Lorentzians with variable offset, a common Lorentzian FWHM, one broad Gaussian, and baseline
threeLorentzian_oneGaussian_oneFWHM = "   w[2] * (0.5*w[0]) / ( (x     )**2 + (0.5*w[0])**2 ) " \
                                      " + w[3] * (0.5*w[0]) / ( (x-w[6])**2 + (0.5*w[0])**2 ) " \
                                      " + w[4] * (0.5*w[0]) / ( (x-w[7])**2 + (0.5*w[0])**2 ) " \
                                      " + w[5] * np.exp(-(x-w[8])**2/(2*w[1]**2)) " \
                                      " + 0*w[9] "

# Three Lorentzians with variable offset, a common Lorentzian FWHM, one broad Gaussian, and baseline
twoLorentzian_oneGaussian_oneFWHM = "   w[2] * (0.5*w[0]) / ( (x     )**2 + (0.5*w[0])**2 ) " \
                                    " + w[3] * (0.5*w[0]) / ( (x-w[5])**2 + (0.5*w[0])**2 ) " \
                                    " + w[4] * np.exp(-(x-w[6])**2/(2*w[1]**2)) " \
                                    " + w[7] "

def make_theta_twoLorentzian_oneGaussian_oneFWHM(params_filepath):
    if params_filepath.split('.')[-1] == 'xls':
        p = pd.read_excel(params_filepath, index_col=0)
    elif params_filepath.split('.')[-1] == 'csv':
        p = pd.read_csv(params_filepath, index_col=0)
    var_names = ['Γ', 'σ', 'a1', 'a2', 'gaussian amp', 'energy offset', 'gaussian offset', 'y0']
    return {                     #   Γ,            σ,           a1,           a2,    gauss amp,           o1,       gauss o,           y0
        'init': np.array([p.iloc[0][0], p.iloc[0][3], p.iloc[0][4], p.iloc[0][5], p.iloc[0][7], p.iloc[0][8], p.iloc[0][10], p.iloc[0][11]]),
          'lb': np.array([p.iloc[1][0], p.iloc[1][3], p.iloc[1][4], p.iloc[1][5], p.iloc[1][7], p.iloc[1][8], p.iloc[1][10], p.iloc[1][11]]),
          'ub': np.array([p.iloc[2][0], p.iloc[2][3], p.iloc[2][4], p.iloc[2][5], p.iloc[2][7], p.iloc[2][8], p.iloc[2][10], p.iloc[2][11]]),
    }, var_names

# Two Lorentzians with the same Γ and one acoustic spectral density
twoLorentzian_oneJ = "   w[2] * (0.5*w[0]) / ( (x     )**2 + (0.5*w[0])**2 ) " \
                     " + w[3] * (0.5*w[0]) / ( (x-w[5])**2 + (0.5*w[0])**2 ) " \
                     " + w[4]*np.abs(x)**w[6]*np.exp(-x/w[1])*np.heaviside(x, 1)" \
                     " + w[7] "

twoLorentzian_oneJ_norm = "  w[0]/2 * (w[2] * (0.5*w[0]) / ( (x     )**2 + (0.5*w[0])**2 ) " \
                          " + w[3] * (0.5*w[0]) / ( (x-w[5])**2 + (0.5*w[0])**2 )) " \
                          " + w[4]*np.abs(x)**w[6]*np.exp(-x/w[1])*np.heaviside(x, 1)" \
                          " + w[7] "


def make_theta_twoLorentzian_oneJ(params_filepath):
    if params_filepath.split('.')[-1] == 'xls':
        p = pd.read_excel(params_filepath, index_col=0)
    elif params_filepath.split('.')[-1] == 'csv':
        p = pd.read_csv(params_filepath, index_col=0)
    var_names = ['Γ', 'ωc', 'a1', 'a2', 'λ', 'energy offset', 'ohmic power', 'y0']
    return {  #                             Γ,           ωc,           a1,           a2,             λ,           o1,   ohmic power,           y0
               'init': np.array([p.iloc[0][0], p.iloc[0][12], p.iloc[0][4], p.iloc[0][5], p.iloc[0][7], p.iloc[0][8], p.iloc[0][13], p.iloc[0][11]]),
               'lb'  : np.array([p.iloc[1][0], p.iloc[1][12], p.iloc[1][4], p.iloc[1][5], p.iloc[1][7], p.iloc[1][8], p.iloc[1][13], p.iloc[1][11]]),
               'ub'  : np.array([p.iloc[2][0], p.iloc[2][12], p.iloc[2][4], p.iloc[2][5], p.iloc[2][7], p.iloc[2][8], p.iloc[2][13], p.iloc[2][11]]),
           }, var_names


                                                                  # la1    g1  la2  la3  lo1  lo2  ga1    s1  go1    y0


def make_theta_twoLorentzian_oneJ_gaussConv(params_filepath):
    if params_filepath.split('.')[-1] == 'xls':
        p = pd.read_excel(params_filepath, index_col=0)
    elif params_filepath.split('.')[-1] == 'csv':
        p = pd.read_csv(params_filepath, index_col=0)
    var_names = ['Γ', 'ωc', 'a1', 'a2', 'λ', 'energy offset', 'ohmic power', 'y0', 'σ']
    return {  #                             Γ,           ωc,           a1,           a2,             λ,           o1,   ohmic power,           y0             σ
               'init': np.array([p.iloc[0][0], p.iloc[0][12], p.iloc[0][4], p.iloc[0][5], p.iloc[0][7], p.iloc[0][8], p.iloc[0][13], p.iloc[0][11], p.iloc[0][3]]),
               'lb'  : np.array([p.iloc[1][0], p.iloc[1][12], p.iloc[1][4], p.iloc[1][5], p.iloc[1][7], p.iloc[1][8], p.iloc[1][13], p.iloc[1][11], p.iloc[1][3]]),
               'ub'  : np.array([p.iloc[2][0], p.iloc[2][12], p.iloc[2][4], p.iloc[2][5], p.iloc[2][7], p.iloc[2][8], p.iloc[2][13], p.iloc[2][11], p.iloc[2][3]]),
           }, var_names


                                                                  # la1    g1  la2  la3  lo1  lo2  ga1    s1  go1    y0


theta_TripleLorentzian_OneGaussian_SingleGamma = {'init': np.array([0.1, 1e-1, 0.1, 0.1, 0.5, 1.5, 0.0, 1e+1, 0.0, 0.01]),
                                                    'lb': np.array([0.0, 1e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 1e+0, 0.0, 0.00]),
                                                    'ub': np.array([2.0, 1e-1, 2.0, 2.0, 1.0, 2.0, 2.0, 1e+3, 3.0, 0.10])
                                                 }

# Single Voight profile
Voigt = "w[0]*(w[1]*(np.exp(-(x-w[2])**2/(2*w[3]**2)))+(1-w[1])*(1/2)*w[4]/((x-w[2])**2+((1/2)*w[4])**2))+0*w[5]"

                               #   A    B     C    D    E    y0
theta_Voigt = {'init': np.array([0.2, 0.5,  0.0,  20,  20, 0.01]),
                 'lb': np.array([0.1, 0.0,  0.0,   1,   1, 0.00]),
                 'ub': np.array([1.0, 1.0, 20.0, 100, 100, 0.05])
               }

GaussianLorentzian = "w[0] * np.exp(-(x)**2/(2*w[1]**2)) + w[2] * (0.5*w[3]) / ( (x-w[4])**2 + (0.5*w[3])**2 ) + 0*w[5]"

                                            #  a1   s1   a2   g1   o    y0
theta_GaussianLorentzian = {'init': np.array([0.2,  20, 0.2,  10,  0, 0.01]),
                              'lb': np.array([0.0,  10, 0.0,  10,  0, 0.00]),
                              'ub': np.array([1.0, 100, 1.0, 100, 50, 0.05])
                           }




def gaussian(w, weg, a, c):
    return a*np.exp(-(w-weg)**2/(2*c**2))


def lorentzian(w, weg, a, c):
    return (a)*c**2/((w-weg)**2+c**2)


def cdf(w, weg, s):
    return (1 + erf(s*(w-weg)/np.sqrt(2)))/2


def skewed_gaussian(w, weg, a, c, s):
    return gaussian(w, weg, a, c) * cdf(w, weg, s)

# def skewed_gaussian(w, weg, a, c, s):
#     return gaussian(w, weg, a, c) * cdf(w, weg, s)

eV2cm = 8065.54429
cm2eV = 1/eV2cm

def ζ_to_energy(w):
    """
    Convert the energy difference axis of a spectral correlation into it's corresponding energy of a spectrum
    """
    n = (len(w)-1)/2
    delta = w[1] - w[0] # energy difference
    return np.linspace(-n/2*delta, n/2*delta, ceil(n)+1)


def energy_to_ζ(w):
    """
    Convert the energy axis of a spectrum into it's corresponding energy difference axis in a spectral correlation
    """
    n = (len(w)-1)*2
    delta = w[1] - w[0] # energy difference
    return np.linspace(-n/2*delta, n/2*delta, ceil(n)+1)


def fourier_ζ_to_cm(x):
    n = len(x)
    d = np.array((max(x) - min(x))/(n-1))
    return np.fft.fftshift(np.fft.fftfreq(n, d)) * cm2eV * 1000

def stage_mm_to_ζ(x):
    # some constants
    N = len(x)
    path_length_difference = 0.1*x  # NOTE: This is where we convert to path length difference space in cm.
    delta = (max(path_length_difference) - min(path_length_difference)) / (N - 1)
    return np.fft.fftshift(np.fft.fftfreq(N, delta)) * cm2eV * 1000  # in meV

def fourier_ζ_to_ps(x):
    delay_cm = fourier_ζ_to_cm(x)
    return delay_cm*1e-2*1e12/299792458

def ps_to_mm(x):
    return 0.2998*x


# if __name__ == '__main__':
#     param_path = '/Users/andrewproppe/Bawendi_Lab/python/PCFS/params/'
#     spectrum_theta, spectrum_vars = make_theta_twoLorentzian_oneJ(param_path + 'PCFSinterferogram_fit_params_CdSe.xls')
#
#     x = np.linspace(-3, 3, 300)
#     y = make_spectrum(x=x, function=twoLorentzian_oneJ, theta=spectrum_theta['init'])
#     # J = x*1*np.exp(-x/0.2)*np.heaviside(x, 1)
#
#     plt.plot(x, y)
#     # plt.plot(x, J)