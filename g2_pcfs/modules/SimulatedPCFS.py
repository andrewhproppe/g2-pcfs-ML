from g2_pcfs.visualization.AP_figs_funcs import *
import numpy as np
import seaborn as sns
import time
import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch.nn.functional import conv2d
from g2_pcfs.modules.conversions import *
from sklearn.preprocessing import normalize
from scipy import signal
from g2_pcfs.modules.PCFS import energy_to_ζ, spec_corr_to_interferogram_FFT, fourier_ζ_to_ps, threeLorentzian_oneFWHM, make_theta_threeLorentzian_oneFWHM, twoLorentzian_oneJ_norm, make_theta_twoLorentzian_oneJ

import pickle
if not matplotlib.get_backend() == 'MacOSX':
    matplotlib.use('MacOSX')
decimal_precision = 1e-12

"""
Version changes:
- Spectrum, spectral correlation, and energy axis are first generated with a finer energy axis, before being downsampled into array that matches
  experimental input. This is to avoid the interferogram changing dramatically with nstage and δ_max
"""

def gauss_peak(x, x0, gauss_fwhm):
    return (gauss_fwhm/(2*np.sqrt(2*np.log(2))))*np.exp(-(4*np.log(2)/(gauss_fwhm**2))*(x-x0)**2)

def poisson_sample_log(y, df, amp=1):
    df = (df/min(df))*amp
    poisson_y = np.random.poisson(y*df, size=len(y))
    poisson_y = poisson_y / df
    return poisson_y

def normalize_torch(x):
    x_normed = x / x.max(1, keepdim=True)[0]
    return x_normed

def make_theta_wiener_diffusion(params_filepath):
    if params_filepath.split('.')[-1] == 'xls':
        p = pd.read_excel(params_filepath, index_col=0)
    elif params_filepath.split('.')[-1] == 'csv':
        p = pd.read_csv(params_filepath, index_col=0)
    var_names = ['weg', 'g2_amp', 'σ_max', 'p', 'α', 'r']
    return {                   #   weg,       g2_amp,        σ_max,            p,            α,            r
        'init': np.array([p.iloc[0][0], p.iloc[0][1], p.iloc[0][2], p.iloc[0][3], p.iloc[0][4], p.iloc[0][5]]),
          'lb': np.array([p.iloc[1][0], p.iloc[1][1], p.iloc[1][2], p.iloc[1][3], p.iloc[1][4], p.iloc[1][5]]),
          'ub': np.array([p.iloc[2][0], p.iloc[2][1], p.iloc[2][2], p.iloc[2][3], p.iloc[2][4], p.iloc[2][5]]),
    }, var_names

def make_theta_diffusion(params_filepath):
    if params_filepath.split('.')[-1] == 'xls':
        p = pd.read_excel(params_filepath, index_col=0)
    elif params_filepath.split('.')[-1] == 'csv':
        p = pd.read_csv(params_filepath, index_col=0)
    var_names = ['weg', 'g2_amp', 'σ_max', 'p', 'p_flat', 'α', 'r', 'flat_max', 'ε']
    return {                   #   weg,       g2_amp,        σ_max,            p,       p_flat,            α,            r,     flat_max,             ε
        'init': np.array([p.iloc[0][0], p.iloc[0][1], p.iloc[0][2], p.iloc[0][3], p.iloc[0][4], p.iloc[0][5], p.iloc[0][6], p.iloc[0][7], p.iloc[0][8]]),
          'lb': np.array([p.iloc[1][0], p.iloc[1][1], p.iloc[1][2], p.iloc[1][3], p.iloc[1][4], p.iloc[1][5], p.iloc[1][6], p.iloc[1][7], p.iloc[1][8]]),
          'ub': np.array([p.iloc[2][0], p.iloc[2][1], p.iloc[2][2], p.iloc[2][3], p.iloc[2][4], p.iloc[2][5], p.iloc[2][6], p.iloc[2][7], p.iloc[2][8]]),
    }, var_names

def make_lineshape(x, function, theta):
    w = np.array(theta)
    f = eval(function)
    return f

def compare_g2s(sdot, dot, delays, show_g2=True, show_xcorr=False, show_acorr=False, xlim=(1e6, 1e11), ylim=(0.5, 1.1)):
    colors = np.array(sns.color_palette("icefire", len(delays)))
    Tot = len(delays)
    if len(delays) < 3:
        Cols = len(delays)
        Rows = 1
    elif len(delays) == 3:
        Cols = 3
        Rows = 1
    elif len(delays) == 4:
        Cols = 2
        Rows = 2
    elif len(delays) == 5 or len(delays) == 6:
        Cols = 3
        Rows = 2
    else:
        Cols = 4
        Rows = Tot // Cols
        Rows += Tot % Cols
    if len(delays) > 1:
        fig, axes = plt.subplots(figsize=(4 * Cols, 3 * Rows), dpi=150, nrows=Rows, ncols=Cols, squeeze=False)
        axes = axes.flatten()
        for i, d in enumerate(delays):
            idx = find_nearest(dot.optical_delay, d)
            if show_g2:
                axes[i].semilogx(sdot.τ, dot.g2[:, idx], label='Data (%.1f ps)' % d, color=colors[i])
            if show_xcorr:
                axes[i].semilogx(sdot.τ, dot.cross_correlations[idx], color=colors[i], label='Data')
            if show_acorr:
                axes[i].semilogx(sdot.τ, dot.auto_correlations[idx], color=colors[i], label='Data')
            axes[i].semilogx(sdot.τ, sdot.g2[:, idx], label='Simulated', color=colors[i] * 0.75)
            axes[i].set_xscale('log')
    else:
        plt.figure(figsize=(4, 4), dpi=150)
        idx = find_nearest(dot.optical_delay, delays[0])
        i = 0
        if show_g2:
            plt.semilogx(sdot.τ, dot.g2[:, idx], label='Data', color=colors[i])
        if show_xcorr:
            plt.semilogx(sdot.τ, dot.cross_correlations[idx], color=colors[i], label='Data')
        if show_acorr:
            plt.semilogx(sdot.τ, dot.auto_correlations[idx], color=colors[i], label='Data')
        plt.semilogx(sdot.τ, sdot.g2[:, idx], label='Simulated', color=colors[i] * 0.75)
        plt.xscale('log')
    dress_fig(tight=True, xlabel='$\\tau$ (ps)', ylabel='$g^{(2)}(\\tau)$', ylim=ylim, xlim=xlim)

def merge_thetas(Θ1, Θ2, var_names1, var_names2):
    ds = [Θ1, Θ2]
    theta = {}
    for k in Θ1.keys():
      theta[k] = np.concatenate(list(d[k] for d in ds))

    var_names = var_names1 + var_names2
    return theta, var_names

def make_theta_pcfs(param_path):
    theta_frame = pd.read_excel(param_path)
    var_names = np.array(theta_frame.columns[1:-1], dtype='str')
    pre_theta = theta_frame.to_numpy()
    new_theta = {}
    new_theta['init'] = np.array(pre_theta[0][1:-1], dtype=np.float32)
    new_theta['lb'] = np.array(pre_theta[1][1:-1], dtype=np.float32)
    new_theta['ub'] = np.array(pre_theta[2][1:-1], dtype=np.float32)
    return new_theta, var_names

def np_to_torch(arr, device):
    return torch.from_numpy(np.array(arr)).to(device)

class SimulatedPCFS(object):
    def __init__(self, time_bounds=[1e4, 1e13], nstage=100, max_δ=100, dither_distance=514, dither_period=20, lag_precision=7, dot=None, ncolors=12, use_torch=False):
        self.use_torch = use_torch
        if self.use_torch:
            if torch.cuda.is_available():
                torch.cuda.current_device()
                torch.cuda.device(0)
                torch.cuda.device_count()
                torch.cuda.get_device_name(0)
                self.device = torch.device('cuda:0')
            else:
                self.device = 'cpu'

        if dot is not None:
            """ Load in an experimental dot.pickle file from a PCFS experiment to extract real stage delays, nstage, max_δ, and tau array parameters """
            self.nstage = len(dot.stage_positions)
            self.max_δ = max(dot.optical_delay)
            self.dot_delay = dot.optical_delay # need the actual stage positions if nonlinear stepping was used
            self.make_tau(dot.time_bounds, dot.lag_precision)
            self.τ = dot.tau

        else:
            self.nstage = nstage
            self.max_δ = max_δ
            self.time_bounds = time_bounds
            self.lag_precision = lag_precision
            self.make_tau(time_bounds, lag_precision)
            self.dot_delay = None

        self.dither_distance = dither_distance
        self.dither_period = dither_period
        self.V = 2*(dither_distance*1e-9)/dither_period

        self.spec_corr_poisson = None
        self.spec_corr_wiener = None
        self.spec_corr_hybrid = None
        # self.colors = np.array(sns.color_palette("Spectral", ncolors))
        # self.colors = np.array(sns.color_palette("icefire", ncolors))
        color = np.array(sns.color_palette("icefire", ncolors+2)); color = np.delete(color, [4, 5], axis=0)
        self.colors = color
        # self.colors = np.array(sns.color_palette("viridis", ncolors))

        if self.use_torch:
            self.τ = np_to_torch(self.τ, self.device)
            self.df = np_to_torch(self.df, self.device)


    def make_delay(self, delay_pos=None):
        if delay_pos is None:
            delay_pos = np.linspace(0, self.max_δ, self.nstage)
        delay_neg = np.flip(-1 * delay_pos)
        delay_neg = np.delete(delay_neg, -1)
        delay_tot = np.concatenate((delay_neg, delay_pos))
        self.δ = delay_tot
        self.δ_pos = delay_pos
        self.δ_neg = delay_neg
        self.i = int(len(delay_tot)/2)
        self.δ_cm = 1 / ps_to_wn(self.δ)

    def make_delay_nonlinear(self, nonlinear_args):
        e = np.logspace(start=nonlinear_args[0], stop=nonlinear_args[1], base=nonlinear_args[2], num=self.nstage, endpoint=True) - 1
        g = np.gradient(e)
        g[0] = 0
        g = g * self.max_δ / sum(g)
        g = g.round(4).transpose()
        δ_pos = np.cumsum(g)
        δ_neg = np.flip(-1 * δ_pos)
        δ_neg = np.delete(δ_neg, -1)
        δ_tot = np.concatenate((δ_neg, δ_pos))
        self.δ = δ_tot
        self.δ_pos = δ_pos
        self.δ_neg = δ_neg
        self.i = int(len(δ_tot)/2)
        self.δ_cm = 1 / ps_to_wn(self.δ)

    def make_tau(self, time_bounds, lag_precision):
        start_time, stop_time = time_bounds

        '''create log 2 spaced lags'''
        cascade_end = int(np.log2(stop_time)) # cascades are collections of lags  with equal bin spacing 2^cascade
        nper_cascade =  lag_precision # number of equal
        a = np.array([2**i for i in range(1,cascade_end+1)])
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

        self.τ = lags
        self.df = division_factor

    def make_spectrum(self, weg, lineshape, theta, nbins=501, emax=5, norm=True, zero_baseline=False):
        """
        - Generate spectrum with a given number of bins and energy range. This energy array gives the corresponding ζ and δ arrays, which
         are later downsampled to match realistic experimental conditions
        - If the energy axis for the spectrum was instead generated by starting from δ, converting to ζ, and then converting to energy, the resolution
        of the spectrum will vary too much with nstage and δ_max --> causes the interferogram and g2s to look completely different depending on those
        parameters
        """
        self.lineshape = lineshape
        self.theta_lineshape = theta
        self.e = np.linspace(-emax, emax, nbins)
        self.ζ = energy_to_ζ(self.e)
        self.δ = fourier_ζ_to_ps(self.ζ)
        self.δ_cm = 1 / ps_to_wn(self.δ)
        self.i = int(len(self.δ)/2) # an index to find the white fringe if using symmetric interferograms
        # self.ζ = self.ζ
        # self.e = ζ_to_energy(self.ζ)
        self.spectrum = make_lineshape(self.e, lineshape, theta)
        if zero_baseline:
            self.spectrum = self.spectrum - min(self.spectrum)
        self.homogeneous_spec_corr = np.correlate(self.spectrum, self.spectrum, mode='full')
        if norm:
            self.homogeneous_spec_corr = self.homogeneous_spec_corr/max(self.homogeneous_spec_corr)

        self.w0 = eV_to_angfreq(weg / 1000)

        if self.use_torch:
            self.homogeneous_spec_corr = np_to_torch(self.homogeneous_spec_corr, self.device)
            self.ζ = np_to_torch(self.ζ, self.device)
            self.δ_cm = np_to_torch(self.δ_cm, self.device)
            self.δ = np_to_torch(self.δ, self.device)
            self.e = np_to_torch(self.e, self.device)

    """ Wiener broadening """
    @staticmethod
    def wiener_diffusion(ζ, τ, α, σ_max, p=1):
        σ = (2*α**2*τ**p)
        return (1/(π*σ))*np.exp(-ζ**2/(2*α**2*τ**p))

    @staticmethod
    def sigmoid_saturation_wiener_diffusion(ζ, τ, α, σ_max, p=1, use_torch=False):
        if use_torch:
            fn = torch
        else:
            fn = np
        σ = (2*α**2*τ**p)
        σ_new = σ_max/(1+fn.exp(-σ))
        σ_new = σ_new - σ_max/2
        σ_new = σ_new*2
        return (1/(π*σ))*fn.exp(-ζ**2/(σ_new))

    @staticmethod
    def inverse_saturation_wiener_diffusion(ζ, τ, α, σ_max, p=1):
        σ = (2*α**2*τ**p)
        σ_new = σ_max/(1+(1/σ))
        return (1/(π*σ))*np.exp(-ζ**2/(σ_new))

    """ Poisson broadening """
    @staticmethod
    def poisson_diffusion(r, τ, δ, σ, p, ε=None):
        σ = eV_to_wn(σ/1000) # convert σ from meV to eV to cm^-1, to cancel out stage delay in cm
        res = np.exp(-r*τ**p*(1-np.exp(-2*pi**2*σ**2*δ**2)))
        ft  = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(res))))
        return ft

    @staticmethod
    def saturating_poisson_diffusion(r, τ, δ, σ, p, ε=20):
        σ = eV_to_wn(σ/1000) # convert σ from meV to eV to cm^-1, to cancel out stage delay in cm
        λ = r*(τ/ε)**p
        λs = ε/(1+(1/λ))
        res = np.exp(-λs*(1-np.exp(-2*pi**2*σ**2*δ**2)))
        ft = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(res))))
        return ft

    @staticmethod
    def saturating_poisson_diffusion_torch(r, τ, δ, σ, p, ε=20):
        σ = eV_to_wn(σ/1000) # convert σ from meV to eV to cm^-1, to cancel out stage delay in cm
        λ = r*(τ/ε)**p
        λs = ε/(1+(1/λ))
        res = torch.exp(-λs*(1-torch.exp(-2*pi**2*σ**2*δ**2)))
        ft = torch.real(torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(res))))
        return ft

    @staticmethod
    def flat_diffusion(ζ, τ, α, ι_min, ι_max, p=1, add_δ=True):
        arg = (2*α**2*τ**p)
        ι   = (ι_max-ι_min)/(1+np.exp(-arg))
        ι   = ι - ι_max/2 + ι_min
        ι   = ι*2
        if add_δ:
            δ = np.exp(-ζ**2/(1e-5))
        else:
            δ = 0*ζ
        return δ+ι


    def make_flat(self, α, p, ι_max, ι_min, add_δ=False, mode='Saturate'):
        self.flat = np.zeros((len(self.τ), len(self.ζ)))

        if mode == 'Normal':
            flat_mode = SimulatedPCFS.flat_diffusion
        elif mode == 'Saturate':
            flat_mode = SimulatedPCFS.flat_diffusion

        ζ, Tau = np.meshgrid(self.ζ, self.τ, sparse=True)
        if self.use_torch:
            self.flat = np_to_torch(self.flat, self.device)
            ζ = np_to_torch(ζ, self.device)
            Tau = np_to_torch(Tau, self.device)
        self.flat = flat_mode(ζ=ζ, τ=Tau, α=α, ι_max=ι_max, ι_min=ι_min, p=p, add_δ=add_δ)


    def make_wiener(self, α, p, σ_max, mode='Saturate', norm=False):
        self.wiener = np.zeros((len(self.τ), len(self.ζ)))
        ζ, Tau = np.meshgrid(self.ζ, self.τ, sparse=True)

        if mode == 'Normal':            # Limitless broadening
            wiener_mode = SimulatedPCFS.wiener_diffusion
        elif mode == 'Saturate':            # Sigmoidally saturating broadening
            wiener_mode = SimulatedPCFS.sigmoid_saturation_wiener_diffusion
        elif mode == 'Inverse Saturate':     # Inverted saturating broadening
            wiener_mode = SimulatedPCFS.inverse_saturation_wiener_diffusion

        if self.use_torch:
            ζ = np_to_torch(ζ, self.device)
            Tau = np_to_torch(Tau, self.device)
            self.wiener = np_to_torch(self.wiener, self.device)

        self.wiener = wiener_mode(ζ=ζ, τ=Tau, α=α, σ_max=σ_max, p=p)
        if norm:
            if self.use_torch:
                self.wiener = normalize_torch(self.wiener)
            else:
                self.wiener = normalize(self.wiener, axis=1, norm='max')


    def make_poisson(self, r, σ_max, p, ε=None, mode='Saturate', norm=False):
        if ε is None:
            ε = σ_max*1000
        Delay, Tau = np.meshgrid(self.δ_cm, self.τ, sparse=True)
        if self.use_torch:
            Delay = np_to_torch(Delay, self.device)
            Tau = np_to_torch(Tau, self.device)
        if mode == 'Normal':
            poisson_mode = SimulatedPCFS.poisson_diffusion
        elif mode == 'Saturate':
            if self.use_torch:
                poisson_mode = SimulatedPCFS.saturating_poisson_diffusion_torch
            else:
                poisson_mode = SimulatedPCFS.saturating_poisson_diffusion
        self.poisson = poisson_mode(r, Tau, Delay, σ_max, p, ε)
        if norm:
            if self.use_torch:
                self.poisson = normalize_torch(self.poisson)
            else:
                self.poisson = normalize(self.poisson, axis=1, norm='max')

    def make_linear(self, σ, p, type='Correlation', norm=True):
        if type == 'Spectral':
            axis = self.e
        elif type == 'Correlation':
            axis = self.ζ

        self.linear = []
        C = np.linspace(1e-4, σ, len(self.τ))
        # C = (σ*np.log(self.τ))**p
        for i in range(0, len(self.τ)):
            gauss = gauss_peak(axis, 0, C[i])
            if norm:
                gauss = gauss/max(gauss)
            self.linear.append(gauss)

    def make_hybrid(self, types, coeffs):
        self.hybrid = np.zeros((len(self.τ), len(self.ζ)))
        if self.use_torch:
            self.hybrid = np_to_torch(self.hybrid, self.device)
        for i, type in enumerate(types):
            self.hybrid = self.hybrid + coeffs[i]*getattr(self, type)

    # @jit
    def convolve(self, diffusion='wiener', norm=True, add_baseline=False):
        self.spec_corr = []

        if diffusion == 'wiener':
            d = self.wiener
        elif diffusion == 'poisson':
            d = self.poisson
        elif diffusion == 'linear':
            d = self.linear
        elif diffusion == 'flat':
            d = self.flat
        elif diffusion == 'hybrid':
            d = self.hybrid

        if self.use_torch:
            dt = d
            dt = dt.unsqueeze(0).unsqueeze(0).to(self.device)
            ht = self.homogeneous_spec_corr
            ht = ht.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)
            conv = conv2d(dt, ht, padding='same').squeeze(0).squeeze(0)
            self.spec_corr = conv.to('cpu')
        else:
            flat_convolve = signal.fftconvolve(d.flatten(), self.homogeneous_spec_corr, mode='same')
            self.spec_corr = flat_convolve.reshape(d.shape[0], d.shape[1])

        self.spec_corr[np.where(self.spec_corr < 1e-7)] = 0
        if add_baseline:
            self.spec_corr = self.spec_corr + self.theta_lineshape[-1]
        if norm:
            if self.use_torch:
                self.spec_corr = normalize_torch(self.spec_corr)
            else:
                self.spec_corr = normalize(self.spec_corr, axis=1, norm='max')

    def make_interferogram(self, norm=True):
        if self.use_torch:
            self.I = torch.real(torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(self.spec_corr))))
        else:
            self.I = spec_corr_to_interferogram_FFT(self.spec_corr)
        self.I[np.where(self.I < 1e-7)] = 0
        if norm:
            if self.use_torch:
                self.I = normalize_torch(self.I)
            else:
                self.I = normalize(self.I, axis=1, norm='max')

    def make_subΙ(self, t1=1e1, t2=1e20, norm=True):
        self.ti1 = find_nearest(t1, self.τ)
        self.ti2 = find_nearest(t2, self.τ)
        self.subI = self.I[self.ti1:self.ti2, :]
        if norm:
            self.subI = normalize(self.subI, axis=1, norm='max')

    def make_subg2(self, t1=1e7, t2=1e11):
        self.ti1 = find_nearest(t1, self.τ)
        self.ti2 = find_nearest(t2, self.τ)
        self.subg2 = self.g2[self.ti1:self.ti2, :]

    def plot_subI(self, clim=([0, 1]), figsize=(4, 4)):
        plt.figure(dpi=150, figsize=figsize)
        plt.contourf(self.δ, self.τ[self.ti1:self.ti2], self.subI, 50)
        plt.clim(clim)
        plt.yscale('log')
        plt.colorbar()
        dress_fig(tight=True, xlabel='$\\delta$ (ps)', ylabel='$\\tau$ (ps)')

    def plot_subg2(self, clim=([0.5, 1]), figsize=(4, 4)):
        plt.figure(dpi=150, figsize=figsize)
        plt.contourf(self.δ, self.τ[self.ti1:self.ti2], self.subg2, 50)
        plt.clim(clim)
        plt.yscale('log')
        plt.colorbar()
        dress_fig(tight=True, xlabel='$\\delta$ (ps)', ylabel='$\\tau$ (ps)')

    def downsample(self, symmetric=False, nonlinear_steps=False, nonlinear_args=[0, 1, 10]):
        """
        - After generating spectra and spectral correlations using a finer energy axis, interferograms are downsampled to match real experimental
        conditions given by max_δ and nstage
        - Only interferograms are downsampled for faster performance
        """
        delay_upsampled = self.δ

        if nonlinear_steps:
            self.make_delay_nonlinear(nonlinear_args)
        else:
            self.make_delay(self.dot_delay) # make real experimental delay again

        interferogram_ds = np.zeros((len(self.τ), len(self.δ)))
        for i in range(len(self.I[:, 0])):
            interferogram_ds[i, :] = np.interp(self.δ, delay_upsampled, self.I[i, :])
        self.I = interferogram_ds

        g2_ds = np.zeros((len(self.τ), len(self.δ)))
        for i in range(len(self.g2[:, 0])):
            g2_ds[i, :] = np.interp(self.δ, delay_upsampled, self.g2[i, :])
        self.g2 = g2_ds

        self.symmetric = symmetric
        if not symmetric:
            self.I = self.I[:, self.i:]
            self.g2 = self.g2[:, self.i:]
            self.δ = self.δ[self.i:]


    def make_g2(self, amp=0.5):
        Tau, Delay = np.meshgrid(self.τ, self.δ, sparse=True)
        if self.use_torch:
            Tau = np_to_torch(Tau, self.device)
            # Delay = np_to_torch(Delay, self.device)
        # self.g2 = 1 - amp*np.cos(2*self.w0*self.V*Tau.T*1e-12/c)*self.I
        self.g2 = (1 - amp*np.cos(2*self.w0*self.V*Tau.T*1e-12/c)*self.I)

    def plot_contour(self, t1=0, t2=1e20, type='interferogram', figsize=(4, 4), dpi=150, colorbar=True):
        plt.figure(figsize=figsize, dpi=dpi)
        self.ti1 = find_nearest(t1, self.τ)
        self.ti2 = find_nearest(t2, self.τ)
        int = self.I[self.ti1:self.ti2, :]
        g2 = self.g2[self.ti1:self.ti2, :]
        delay = self.δ
        tau = self.τ[self.ti1:self.ti2]
        if type == 'interferogram':
            plt.contourf(delay, tau, int, 50)
        elif type == 'g2':
            plt.contourf(delay, tau, g2, 50)
        plt.yscale('log')
        dress_fig(tight=True, xlabel='$\\delta$ (ps)', ylabel='$\\tau$ (ps)')
        if colorbar:
            plt.colorbar()


    def plot_spectrum(self, figsize=(4, 4), dpi=150, xlim=[-3, 3], type='spectrum', fixframe=False, legend=False):
        if fixframe:
            fig = make_fig(figsize, dpi)
        else:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        if type == 'spectrum':
            plt.plot(self.e, self.spectrum, label='Spectrum')
        elif type == 'spec corr':
            plt.plot(self.ζ, self.homogeneous_spec_corr, label='Spectral Correlation')
        dress_fig(tight=not fixframe, xlabel='Energy (meV)', ylabel='Intensity (a.u.)', frameon=False, xlim=xlim, legend=False)
        return fig

    def plot_spec_corr(self, taus, type='spec_corr', figsize=(4, 4), dpi=150, xlim=[-3, 3], spacer=0, colors=None, fixframe=False, legend=True):
        if fixframe:
            fig = make_fig(figsize, dpi)
        else:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        if colors is None:
            colors = self.colors
        for i, t in enumerate(taus):
            idx1 = find_nearest(t[0], self.τ)
            idx2 = find_nearest(t[1], self.τ)
            speccorr = getattr(self, type)
            plt.plot(self.ζ, speccorr[idx1:idx2, :].mean(axis=0)+spacer*i, color=colors[i], label=('%1.0e - %1.0e ps') % (t[0], t[1]))
        dress_fig(tight=not fixframe, xlabel='$\\zeta$ (meV)', ylabel='$\it{p}(\\zeta$)', frameon=False, xlim=xlim, legend=legend)
        return fig

    def plot_inteferogram(self, taus, figsize=(4, 4), dpi=150, spacer=0, marker=None, fixframe=False, legend=True):
        if fixframe:
            fig = make_fig(figsize, dpi)
        else:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        for i, t in enumerate(taus):
            idx1 = find_nearest(t[0], self.τ)
            idx2 = find_nearest(t[1], self.τ)
            plt.plot(self.δ, self.I[idx1:idx2, :].mean(axis=0)+spacer*i, ms=2, marker=marker, color=self.colors[i], label=('%1.0e - %1.0e ps') % (t[0], t[1]))
        dress_fig(tight=not fixframe, xlabel='$\\delta$ (ps)', ylabel='Amplitude (a.u.)', legend=legend)
        return fig

    def plot_g2(self, delays, figsize=(4, 4), dpi=150, xlim=[1e5, 1e13], ylim=[0.5, 1.1], fixframe=False):
        if fixframe:
            fig = make_fig(figsize, dpi)
        else:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        for i, t in enumerate(delays):
            idx = find_nearest(t, self.δ)
            plt.plot(self.τ, self.g2[:, idx], color=self.colors[i], label='$\\delta$=%.1f ps' % (self.δ[idx]))
        dress_fig(tight=not fixframe, xlabel='$\\tau$ (ps)', ylabel='$g^{(2)}(\\tau)$', xlim=xlim, ylim=ylim, lgnd_cols=3)
        plt.xscale('log')

        return fig

def generate_pcfs_triplet_wiener(Γ, a1, a2, a3, o1, o2, y0, weg, g2_amp, σ_max, p, α):
    pcfs = SimulatedPCFS(time_bounds=[1e5, 1e11], nstage=30, max_δ=100, lag_precision=7, ncolors=12, dither_period=20, dither_distance=514)
    pcfs.make_spectrum(weg=weg, lineshape=threeLorentzian_oneFWHM, theta=[Γ, a1, a2, a3, o1, o2, y0], nbins=1501, emax=5)
    pcfs.make_wiener(α=α, σ_max=σ_max, p=p, norm=True)
    pcfs.convolve(diffusion='wiener')
    pcfs.make_interferogram()
    pcfs.make_g2(amp=g2_amp)
    pcfs.downsample(nonlinear_steps=True, nonlinear_args=[0, 2, 10])
    return pcfs

def generate_pcfs_triplet_flat(Γ, a1, a2, a3, o1, o2, y0, weg, g2_amp, ι_max, pf, αf):
    pcfs = SimulatedPCFS(time_bounds=[1e5, 1e11], nstage=30, max_δ=100, lag_precision=7, ncolors=12, dither_period=20, dither_distance=514)
    pcfs.make_spectrum(weg=weg, lineshape=threeLorentzian_oneFWHM, theta=[Γ, a1, a2, a3, o1, o2, y0], nbins=1501, emax=5)
    pcfs.make_flat(α=αf, p=pf, ι_max=ι_max, ι_min=0, add_δ=True)
    pcfs.convolve(diffusion='flat')
    pcfs.make_interferogram()
    pcfs.make_g2(amp=g2_amp)
    pcfs.downsample(nonlinear_steps=True, nonlinear_args=[0, 2, 10])
    return pcfs

def generate_pcfs(w, lineshape_type, diffusion_type):
    """
    Works for generating a simulated PCFS experiment where the spectrum and diffusion type can be given as inputs
    :param w: array of parameters for the spectrum and diffusion
    :param lineshape_type: 0 for Lorentzian triplet, 1 for Lorentzian doublet + acoustic sideband
    :param diffusion_type: 0 for Wiener diffusion, 1 for Poisson diffusion
    :return:
    """
    Γ = w[0]; a1 = w[1]; a2 = w[2]; a3 = w[3]; o1 = w[4]; o2 = w[5]; λ = w[6]; ωc = w[7]; ωp = w[8]; y0 = w[9] # Spectrum parameters
    weg = w[10]; g2_amp = w[11]; σ_max = w[12]; p = w[13]; α = w[14]; r = w[15]; ε = w[16]; ι_max = w[17]; pf = w[18]; αf = w[19] # Diffusion parameters

    # pcfs = SimulatedPCFS(time_bounds=[1e5, 1e11], nstage=30, max_δ=50, lag_precision=7, ncolors=5, dither_period=20, dither_distance=514) # for single g2s
    pcfs = SimulatedPCFS(time_bounds=[1e4, 1e11], nstage=100, max_δ=100, lag_precision=5, ncolors=5, dither_period=20, dither_distance=514) #

    if lineshape_type == 0:
        lineshape_function = threeLorentzian_oneFWHM
        spectrum_theta = [Γ, a1, a2, a3, o1, o2, y0]
    elif lineshape_type == 1:
        lineshape_function = twoLorentzian_oneJ_norm
        spectrum_theta = [Γ, ωc, a1, a2, λ, o1, ωp, y0]

    pcfs.make_spectrum(weg=weg, lineshape=lineshape_function, theta=spectrum_theta, nbins=2001, emax=5)

    if diffusion_type == 0:
        pcfs.make_wiener(α=α, σ_max=σ_max, p=p, norm=True)
        pcfs.convolve(diffusion='wiener')
    elif diffusion_type == 1:
        pcfs.make_poisson(r=r, σ_max=σ_max, p=p, ε=ε, norm=True)
        pcfs.convolve(diffusion='poisson')
    elif diffusion_type == 2:
        pcfs.make_flat(α=αf, p=pf, ι_max=ι_max, ι_min=0, add_δ=True)
        pcfs.convolve(diffusion='flat')

    pcfs.make_interferogram()
    pcfs.make_g2(amp=g2_amp)
    # pcfs.downsample(nonlinear_steps=True, nonlinear_args=[0, 2, 10])
    pcfs.downsample()
    return pcfs


# def generate_pcfs_triplet_wiener(Γ, a1, a2, a3, o1, o2, y0, weg, g2_amp, σ_max, p, α, ι_max, pf, αf):
#     pcfs = SimulatedPCFS(time_bounds=[1e5, 1e11], nstage=30, max_δ=100, lag_precision=7, ncolors=12, dither_period=20, dither_distance=514)
#     pcfs.make_spectrum(weg=weg, lineshape=threeLorentzian_oneFWHM, theta=[Γ, a1, a2, a3, o1, o2, y0], nbins=1501, emax=5)
#     pcfs.make_wiener(α=α, σ_max=σ_max, p=p, norm=True)
#     pcfs.make_flat(α=αf, p=pf, ι_max=ι_max, ι_min=0, add_δ=False)
#     pcfs.make_hybrid(types=['wiener', 'flat'], coeffs=[1., 1.])
#     pcfs.convolve(diffusion='hybrid')
#     pcfs.make_interferogram()
#     pcfs.make_g2(amp=g2_amp)
#     pcfs.downsample(nonlinear_steps=True, nonlinear_args=[0, 2, 10])
#     return pcfs


# def generate_pcfs_general(theta):
#     """ Spectrum parameters """
#     Γ = theta[0]; σ = theta[1]; a1 = theta[2]; a2 = theta[3]; a3 = theta[4]; a4 = theta[5]
#     o1 = theta[6]; o2 = theta[7]; o3 = theta[8]; ωc = theta[9]; ωp = theta[10]; y0 = theta[11]
#     """ Diffusion parameters """
#     weg = theta[12]; g2_amp = theta[13]; σ_max = theta[14]; p = theta[15]; p_flat = theta[16]
#     α = theta[17]; r = theta[18]; ι_max = theta[19]; ε = theta[20]
#     """ Generation parameters """
#     lineshape = theta[21]; diffusion = theta[22]; flat_on = theta[23]
#
#     pcfs = SimulatedPCFS(time_bounds=[1e5, 1e11], nstage=75, max_δ=150, lag_precision=8, ncolors=12, dither_period=20, dither_distance=514)
#     pcfs.make_spectrum(weg=weg, lineshape=lineshape_function, theta=lineshape_theta, nbins=2000, emax=10)
#     pcfs.make_wiener(α=α, σ_max=σ_max, p=p, norm=True)
#     pcfs.convolve(diffusion='wiener')
#     pcfs.make_interferogram()
#     pcfs.make_g2(amp=g2_amp)
#     pcfs.downsample()
#     return pcfs

""" DEBUGGING """
if __name__ == '__main__':

    dot_path = '../../dots/'
    dot_name = 'CK174_dot6_full' # perovskite
    # dot_name = 'dotC_20210604.pickle' # Perovskite
    # dot_name = 'dot7_75pos_20mm_1uW.pickle' # CdSe'
    # dot_name = 'dotH_8uW_20210524.pickle' # Perovskite

    pickle_in = open(dot_path+dot_name, "rb"); dot = pickle.load(pickle_in); pickle_in.close()

    # lineshape_function = twoLorentzian_oneGaussian_oneFWHM
    # lineshape_theta, lineshape_vars = make_theta_twoLorentzian_oneGaussian_oneFWHM(param_path+'spectrum_params_dotC.xls')
    # lineshape_function = twoLorentzian_oneGaussian_oneFWHM
    # lineshape_theta, lineshape_vars = make_theta_twoLorentzian_oneGaussian_oneFWHM(param_path+'spectrum_params_dotH_8uW.xls')
    lineshape_function = twoLorentzian_oneJ_norm
    # lineshape_theta, lineshape_vars = make_theta_twoLorentzian_oneJ(param_path+'spectrum_params_CdSe.xls')


    param_path = '../../data/fit_params/'
    lineshape_theta, lineshape_vars = make_theta_twoLorentzian_oneJ(param_path+'spectrum_params_CdSe.xls')
    # lineshape_function = threeLorentzian_oneFWHM
    # lineshape_theta, lineshape_vars = make_theta_threeLorentzian_oneFWHM(param_path+'spectrum_params_CK174_dot6.xls')

    # diffusion_functions = ['wiener']
    # diffusion_theta, diffusion_vars = make_theta_diffusion(param_path+'diffusion_params_dotC.xls')
    # diffusion_theta, diffusion_vars = make_theta_diffusion(param_path+'diffusion_params_CdSe_dot7.xls')
    # diffusion_theta, diffusion_vars = make_theta_diffusion(param_path+'diffusion_params_dotH_8uW.xls')
    diffusion_theta, diffusion_vars = make_theta_diffusion(param_path + 'diffusion_params_CK174_dot6.xls')

    weg    = diffusion_theta['init'][0]
    g2_amp = diffusion_theta['init'][1]
    σ_max  = diffusion_theta['init'][2]
    p      = diffusion_theta['init'][3]
    pf     = diffusion_theta['init'][4]
    α      = diffusion_theta['init'][5]
    r      = diffusion_theta['init'][6]
    fmax   = diffusion_theta['init'][7]
    ε      = diffusion_theta['init'][8]



    tic1 = time.time()
    pcfs = SimulatedPCFS(time_bounds=[1e4, 1e11], nstage=75, max_δ=100, lag_precision=7, ncolors=8, dither_period=20, dither_distance=514,
                         # dot=dot
                         )

    pcfs.make_spectrum(weg=2330, lineshape=lineshape_function, theta=lineshape_theta['init'], nbins=2001, emax=5)
    # pcfs.make_wiener(α=α, σ_max=σ_max, p=p, norm=True)
    # pcfs.make_flat(α=α, p=pf, ι_max=0, ι_min=0, add_δ=False)
    # pcfs.make_poisson(r=r, σ_max=σ_max, p=p, ε=ε, norm=True)
    # pcfs.make_hybrid(types=['wiener', 'flat'], coeffs=[1., 1.])
    # pcfs.make_wiener(α=1e-3, σ_max=1e-1, p=0.5, norm=True)
    pcfs.make_poisson(r=1e-8, σ_max=1e-1, p=1, ε=100, norm=True)

    pcfs.convolve(diffusion='poisson')
    pcfs.make_interferogram()
    pcfs.make_g2(amp=g2_amp)
    pcfs.downsample(nonlinear_steps=True)
    toc = time.time()
    print('Elapsed overall: %.5f' % (toc - tic1))

    # tau_span = ([1e6, 5e6], [1e8, 5e8], [1e9, 5e9], [1e10, 1.1e10], [2e10, 3e10])
    tau_span = ([1e6, 5e6], [1e8, 5e8], [1e9, 5e9], [1e10, 5e10], [1e11, 2e11])
    # tau_span = ([1e7, 5e7], [1e8, 5e8], [1e9, 5e9], [1e10, 3e10])
    #

    set_font_size(8)
    fs = (4, 3)

    pcfs.plot_spectrum(figsize=fs, xlim=[-2, 2])
    # # pcfs.plot_spec_corr(figsize=fs, taus=tau_span, type='wiener', xlim=[-2, 2])
    pcfs.plot_spec_corr(figsize=fs, taus=tau_span, type='spec_corr', xlim=[-2, 2])
    # pcfs.plot_inteferogram(figsize=fs, taus=tau_span, spacer=0)

    #
    # dot.plot_interferogram(tau_span=tau_span)
    # dot.plot_mirror_spectral_corr(tau_span=tau_span)

    pcfs.plot_g2(
        delays=[0, 8, 11.6, 14.7, 18.7, 23.4],
        xlim=[1e6, 1e11],
        ylim=[0.5, 1.1],
        figsize=fs
    )

    # len(pcfs.τ)
    # raise RuntimeError
    # delays = [0, 11, 20]
    # compare_g2s(pcfs, dot, delays, xlim=[1e7, 1e11])
    #
    # color = np.array(sns.color_palette("icefire", 12))
    # color = np.delete(color, [4, 5], axis=0)
    # for i, c in enumerate(color):
    #     plt.plot(pcfs.g2[0]+i/3, color=c)
    #
    # make_fig((3, 3))
    # plt.plot(pcfs.τ, label='τ')
    # plt.plot(pcfs.df, label='Division factor')
    # dress_fig(ylabel='Time (ps)')
    # plt.yscale('log')