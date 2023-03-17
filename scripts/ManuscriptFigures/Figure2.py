from g2_pcfs.visualization.AP_figs_funcs import *
import numpy as np
import time
import sys
from matplotlib import pyplot as plt
from g2_pcfs.modules.PCFS import ζ_to_energy, energy_to_ζ, spec_corr_to_interferogram_FFT, fourier_ζ_to_ps, \
                 threeLorentzian_oneFWHM, make_theta_threeLorentzian_oneFWHM, \
                 twoLorentzian_oneJ, make_theta_twoLorentzian_oneJ
from g2_pcfs.modules.SimulatedPCFS import SimulatedPCFS, make_theta_diffusion
import pickle
decimal_precision = 1e-12

dot_root_path = '../../dots/'
dot_fname = 'dotC_20210604.pickle'
# dot_fname = 'CK174_dot6_full.pickle'
# dot_fname = 'dot7_75pos_20mm_1uW.pickle'
# dot_fname = 'dotH_8uW_20210524.pickle'

sys.path.append('../../g2_pcfs/modules')
pickle_in = open(dot_root_path+dot_fname, "rb"); dot = pickle.load(pickle_in); pickle_in.close()

param_path = '../../data/fit_params/'

# lineshape_function = twoLorentzian_oneGaussian_oneFWHM
# lineshape_theta, lineshape_vars = make_theta_twoLorentzian_oneGaussian_oneFWHM(param_path+'spectrum_params_dotC.xls')
# lineshape_function = twoLorentzian_oneGaussian_oneFWHM
# lineshape_theta, lineshape_vars = make_theta_twoLorentzian_oneGaussian_oneFWHM(param_path+'spectrum_params_dotH_8uW.xls')
# lineshape_function = twoLorentzian_oneJ
# lineshape_theta, lineshape_vars = make_theta_twoLorentzian_oneJ(param_path+'spectrum_params_CdSe.xls')
lineshape_function = threeLorentzian_oneFWHM

lineshape_theta, lineshape_vars = make_theta_threeLorentzian_oneFWHM(param_path+'spectrum_params_CK174_dot6.xls')

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
pcfs = SimulatedPCFS(time_bounds=[1e4, 1e13], nstage=75, max_δ=100, lag_precision=7, ncolors=7, dither_period=20, dither_distance=514,
                     # dot=dot
                     )
pcfs.make_spectrum(weg=2330, lineshape=lineshape_function, theta=lineshape_theta['init'], nbins=2000, emax=5)
# pcfs.make_wiener(α=α, σ_max=σ_max, p=p, norm=True)
# pcfs.make_flat(α=α, p=pf, ι_max=0, ι_min=0, add_δ=False)
# pcfs.make_poisson(r=r, σ_max=σ_max, p=p, ε=ε, norm=True)
# pcfs.make_hybrid(types=['wiener', 'flat'], coeffs=[1., 1.])
pcfs.make_wiener(α=2e-4, σ_max=0.1e-1, p=0.75, norm=True)

# pcfs.make_poisson(r=1e-4, σ_max=1e-1, p=0.25, ε=300, norm=True)
# pcfs.make_poisson(r=1e-5, σ_max=1e-1, p=0.7, ε=1, norm=True)
# pcfs.make_poisson(r=2e-3, σ_max=2e-2, p=0.25, ε=100, norm=True)

pcfs.convolve(diffusion='wiener')
pcfs.make_interferogram()
pcfs.make_g2(amp=g2_amp)
pcfs.downsample(nonlinear_steps=True)
toc = time.time()
print('Elapsed overall: %.5f' % (toc - tic1))

a = 6
b = 7
c = 8
d = 9
e = 10
tau_span = ([1*10**a, 2*10**a], [1*10**b, 2*10**b], [1*10**c, 2*10**c], [1*10**d, 2*10**d], [1*10**e, 2*10**e])

set_font_size(7, lgnd=0)
fs = (2, 1.5)

pcfs.plot_spectrum(figsize=fs, xlim=[-1.5, 1.5], fixframe=True)

# pcfs.plot_wiener(figsize=fs, taus=tau_span, xlim=[-2.0, 2.0], fixframe=True)

pcfs.plot_spec_corr(figsize=fs, taus=tau_span, type='spec_corr', xlim=[-1.5, 1.5], fixframe=True, legend=False)

pcfs.plot_inteferogram(figsize=fs, taus=tau_span, spacer=0, fixframe=True, legend=False)

pcfs.plot_g2(
    delays=[1, 8, 12, 15, 19, 23],
    xlim=[1e6, 1e11],
    ylim=[0.6, 1.1],
    fixframe=True,
    figsize=fs
)


""" Fig. 2f"""
pcfs.plot_g2(
    delays=[16],
    xlim=[1e6, 1e11],
    ylim=[0.6, 1.1],
    fixframe=True,
    figsize=fs
)

from g2_pcfs.pipeline.transforms import poisson_sample_log

g2 = pcfs.g2[:, 29]
g2 = g2/4
seed: int = 10236
rng = np.random.default_rng(seed)
g2_noisy = poisson_sample_log(g2, rng, pcfs.df)
g2_noisy = g2_noisy * 4
plt.plot(pcfs.τ, g2_noisy, label='Noise added', zorder=0)
plt.semilogx(dot.tau, dot.g2_x[2, :] - 0.1, color=[0.2, 0.2, 0.8], label='Expt.')
dress_fig()