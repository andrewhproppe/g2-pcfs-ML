from g2_pcfs.visualization.AP_figs_funcs import *
import time
import sys
from g2_pcfs.modules.SimulatedPCFS import SimulatedPCFS, make_theta_diffusion
from g2_pcfs.modules.PCFS import threeLorentzian_oneFWHM, make_theta_threeLorentzian_oneFWHM
import pickle
decimal_precision = 1e-12

if __name__ == '__main__':

    dot_root_path = '../../dots/'
    # dot_fname = 'CK174_dot6_full.pickle'
    dot_fname = 'dotC_20210604.pickle'
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
                         dot=dot
                         )
    pcfs.make_spectrum(weg=2330, lineshape=lineshape_function, theta=lineshape_theta['init'], nbins=2000, emax=5)

    pcfs.make_poisson(r=1e-5, σ_max=1e-1, p=0.7, ε=1, norm=True,
                      mode='Normal'
                      )

    pcfs.convolve(diffusion='poisson')
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
    fs = (1.3345, 1.1464)

    pcfs.plot_spec_corr(figsize=fs, taus=tau_span, type='spec_corr', xlim=[-1.5, 1.5], fixframe=True, legend=False)

    fig_dims = (2.75, 1.4),

    pcfs.plot_g2(
        delays=[1, 8, 11.6, 14.7, 18.7],
        # delays=[0.43, 0.88, 2.22, 3.2],
        xlim=[1e6, 1e12],
        ylim=[0.6, 1.1],
        fixframe=True,
        figsize=(2.781, 1.4158),
        # figsize=fs
    )