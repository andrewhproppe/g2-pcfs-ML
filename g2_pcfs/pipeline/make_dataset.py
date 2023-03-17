"""
make_dataset.py
This module will implement methods to synthesize the data used for the project.
"""
import pandas as pd
import itertools
import numpy as np
import h5py
from tqdm import tqdm
from warnings import warn
from typing import Tuple

def g2_function(
    x: np.ndarray, w0: float, w1: float, w2: float, w3: float, w4: float, dummy) -> np.ndarray:
    return w0 + w1 * (np.exp(-np.abs(x + w4) / w3) + np.exp(-np.abs(x - w4) / w3) + w2 * np.exp(-np.abs(x) / w3))


def g2_function_monoexp(
    x: np.ndarray, w0: float, w1: float, w2: float, w3: float, w4: float, dummy) -> np.ndarray:
    """
    - Reordered parameters so that amplitude and lifetime are last, to keep order of r -> y0 -> rr the same for monoexp and biexp
    """
    return w1 + w3 * ( w0 * np.exp(-np.abs(x)/w4) + np.exp(-np.abs(x + w2)/w4) + np.exp(-np.abs(x - w2)/w4))


def g2_function_biexp(
    x: np.ndarray, w0: float, w1: float, w2: float, w3: float, w4: float, w5: float, dummy) -> np.ndarray:
    """
    - Contains additional components for a second ampltude and lifetime (w5 and w6)
    """
    return (w3        * (w0 * np.exp(-np.abs(x)/w4) + np.exp(-np.abs(x+1*w2)/w4) + np.exp(-np.abs(x-1*w2)/w4)
                                                    + np.exp(-np.abs(x+2*w2)/w4) + np.exp(-np.abs(x-2*w2)/w4)
                                                    + np.exp(-np.abs(x+3*w2)/w4) + np.exp(-np.abs(x-3*w2)/w4)
                        )
             + (1-w3) * (w0 * np.exp(-np.abs(x)/w5) + np.exp(-np.abs(x+1*w2)/w5) + np.exp(-np.abs(x-1*w2)/w5)
                                                    + np.exp(-np.abs(x+2*w2)/w5) + np.exp(-np.abs(x-2*w2)/w5)
                                                    + np.exp(-np.abs(x+3*w2)/w5) + np.exp(-np.abs(x-3*w2)/w5)
                        )
                        ) + w1


def generate_gridded_spectra(num_points: int, timesteps: int = 1501,  bounds: tuple = None, trange: tuple = (-1.5, 1.5), nexps: int = 1) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
    """
    Function to generate all of the g2 spectra over a linearly spaced grid
    of parameters. This code is vectorized, and so the resulting array shape
    will be (N, N, N, N, T), where N is `num_points`, and T is the number of
    time points used for spectra.

    One thing to note is that I purposely made an effort to downcast to
    single precision; for the deep learning models we don't want to use
    double precision anyway, so storing the precision only to throw it
    out later is a waste of space.
    
    TODO: maybe make the lower/upper boundaries changable

    Parameters
    ----------
    num_points : int
        Number of points for each parameter
    timesteps : int
        Number of timesteps for the spectra

    Returns
    -------
    spectra, parameters
        Returns 
    """
    if timesteps % 2 == 0:
        warn("Timesteps should be odd-numbered!")

    if bounds == None:
        w0_points = np.linspace(0., 1., num_points, dtype=np.float32) # r - center-to-side peak ratio
        w1_points = np.linspace(0.0, 0.075, num_points, dtype=np.float32) # y0 - background constant
        w2_points = np.linspace(0.99, 1.01, num_points, dtype=np.float32) # rr - laser repetition rate
        w3_points = np.linspace(0., 1., num_points, dtype=np.float32) # a1 - amplitude of first lifetime
        w4_points = np.linspace(2e-2, 1.0e-1, num_points, dtype=np.float32) # t1 - first lifetime (in µs)
        w5_points = np.linspace(1e-2, 0.5e-1, num_points, dtype=np.float32) # t2 - second lifetime (in µs)

    else:
        w0_points = np.linspace(bounds[0][0], bounds[0][1], num_points, dtype=np.float32)
        w1_points = np.linspace(bounds[1][0], bounds[1][1], num_points, dtype=np.float32)
        w2_points = np.linspace(bounds[2][0], bounds[2][1], num_points, dtype=np.float32)
        w3_points = np.linspace(bounds[3][0], bounds[3][1], num_points, dtype=np.float32)
        w4_points = np.linspace(bounds[4][0], bounds[4][1], num_points, dtype=np.float32)
        if nexps == 2:
            w5_points = np.linspace(bounds[5][0], bounds[5][1], num_points, dtype=np.float32)

    # generate array for time points
    x = np.linspace(*sorted(trange), timesteps).astype(np.float32)

    # package it together nicely for export
    # generate a grid of data points to vectorize
    # this 
    if nexps == 1:
        func = g2_function_monoexp
        params = (w0_points, w1_points, w2_points, w3_points, w4_points)
    else:
        func = g2_function_biexp
        params = (w0_points, w1_points, w2_points, w3_points, w4_points, w5_points)
    # unpack params into meshgrid generation
    param_grid = np.meshgrid(*params, 1, indexing="ij")

    return func(x, *param_grid), params, x


def create_dataset(filepath: str, num_points: int, timesteps: int, trange: tuple, nexps: int, bounds: tuple = None) -> None:
    """
    Higher level function for generating a dataset and saving it to disk
    in HDF5 format. 

    Parameters
    ----------
    num_points : int
        Number of points for each parameter
    timesteps : int
        Number of timesteps for the spectra
    trange : tuple
        Min and max times for x values
    nexps: int
        Number of exponentials for g2 generation (1 or 2)
    filepath : str
        Name of the file to save the data to, including extension
    bounds: tuple
        Lower and upper bounds for variables in grid
    """


    spectra, params, t = generate_gridded_spectra(num_points=num_points, timesteps=timesteps, trange=trange, nexps=nexps, bounds=bounds)
    with h5py.File(filepath, "a") as h5_data:
        h5_data["spectra"] = spectra
        h5_data["time"] = t
        # save parameters
        for index, param in enumerate(params):
            h5_data.create_dataset(f"parameter_w{index}", data=param)


def param_permutations_gridded(file_path):
    """
    Reads an excel sheet containing the variable lower & upper bounds, and the number of grid points to use for each.
    Uses these values to generate arrays of the values for each variable, before creating every permutation of these values
    :param file_path: file path to .xlsx file
    :return: array of parameter permutations
    """
    df = pd.read_excel(file_path)
    params = []
    for i, (name, data) in enumerate(df.iteritems()):
        if i == 0:
            pass
        elif i > 0:
            arr = np.linspace(data.values[0], data.values[1], int(data.values[2]))
            params.append(arr)
    param_permutations = np.asarray(list(itertools.product(*params)), dtype=np.float32)
    return param_permutations


def param_permutations_random(file_path, n=100):
    """
    Reads an excel sheet containing the variable lower & upper bounds, and the number of random samples to generate
    Uses these values to generate arrays of the values for each variable, before creating every permutation of these values
    :param file_path: file path to .xlsx file
    :param n: number of samples to generate
    :return: array of parameter permutations
    """
    df = pd.read_excel(file_path)
    dfn = df.to_numpy()
    lb = dfn[0][1:].astype(np.float32)
    ub = dfn[1][1:].astype(np.float32)
    params_random = np.random.uniform(low=lb, high=ub, size=(n, len(lb)))

    # p_divider = ((params_random[:, 13])/0.5)**p

    p     = params_random[:, 13]
    α     = params_random[:, 14]
    r     = params_random[:, 15]

    params_random[:, 14] = α/(1*p**(-1*np.log10(α)))
    params_random[:, 15] = r/(1*p**(-1*np.log10(r)))
    # params_random[:, 19] = params_random[:, 19]/(params_random[:, 18]**p)
    # FFFF
    # Make ε dependent on σ_max
    # params_random[:, 16] = params_random[:, 12]*params_random[:, 16]
    # params_random[:, 16] = params_random[:, 12] * 10000

    return params_random


def make_training_g2s(params, pcfs_function, randomize_lineshape=None, randomize_diffusion=None, logt=True):
    """
    - Sequentialy generates simulated PCFS experiments, whose g2s are continuously concatenated
    - One set of parameters is used to generate n g2s (i.e. each experiment generates 50 - 100 g2s), so the parameter array used is
      repeated n times. Redundant but ensures that the DataLoader can always index the parameters and g2s together
    """
    for i, p in enumerate(tqdm(params)):

        if randomize_lineshape is not None:
            r1 = np.random.uniform()
            if r1 >= randomize_lineshape:
                lineshape_type = 0
            elif r1 < randomize_lineshape:
                lineshape_type = 1

        if randomize_diffusion is not None:
            r2 = np.random.uniform()
            if r2 >= randomize_diffusion:
                diffusion_type = 0
            elif r2 < randomize_diffusion:
                diffusion_type = 1

        simPCFS = pcfs_function(p, lineshape_type, diffusion_type)
        if i == 0:
            g2s = simPCFS.g2
            paramstack = np.tile(p, (simPCFS.g2.shape[-1], 1))
        else:
            g2s = np.hstack((g2s, simPCFS.g2))
            paramstack = np.vstack((paramstack, np.tile(p, (simPCFS.g2.shape[-1], 1))))

    g2s = g2s.T
    paramstack = paramstack
    if logt:
        t = np.log(simPCFS.τ)
    else:
        t = simPCFS.τ
    df = simPCFS.df
    nstage = simPCFS.nstage
    return g2s, paramstack, t, df, nstage


def make_training_g2s_2d(params, pcfs_function, randomize_lineshape=None, randomize_diffusion=None, logt=True):
    g2s = []
    paramstack = []
    spectra = []
    for i, p in enumerate(tqdm(params)):

        if randomize_lineshape is not None:
            r1 = np.random.uniform()
            if r1 >= randomize_lineshape:
                lineshape_type = 0
            elif r1 < randomize_lineshape:
                lineshape_type = 1

        if randomize_diffusion is not None:
            r2 = np.random.uniform()
            if r2 >= randomize_diffusion:
                diffusion_type = 0
            elif r2 < randomize_diffusion:
                diffusion_type = 1

        simPCFS = pcfs_function(p, lineshape_type, diffusion_type)
        g2s.append(simPCFS.g2.T)
        paramstack.append(p)
        spectra.append(simPCFS.spectrum)

    # g2s = g2s.T
    # paramstack = paramstack.T
    if logt:
        t = np.log(simPCFS.τ)
    else:
        t = simPCFS.τ
    df = simPCFS.df
    nstage = simPCFS.nstage
    return g2s, paramstack, spectra, t, df, nstage


def augment_training_g2s(g2s, params, index_separation, n):
    new_g2s = np.zeros((n, g2s.shape[-1]))
    new_params = np.zeros((n, params.shape[-1]))
    for a in range(0, n):
        i1 = np.random.randint(0, g2s.shape[0])
        i2 = np.random.randint(0, g2s.shape[0])
        while np.abs(i2-i1) < index_separation:
            # Ensures that g2s are drawn from different PCFS simulations for combination
            i2 = np.random.randint(0, len(g2s[:, 0]))
        new_g2s[a, :] = (g2s[i1, :]+g2s[i2, :])/2
        new_params[a, :] = ((params[i1, :]+params[i2, :])/2)

    g2s = np.vstack((g2s, new_g2s))
    params = np.vstack((params, new_params))
    return g2s, params


def augment_training_g2s_2d(g2s, params, n):
    new_g2s = []
    new_params = []
    for a in range(0, n):
        i1 = np.random.randint(0, g2s.__len__())
        i2 = np.random.randint(0, g2s.__len__())
        while i1 != i2:
            # Ensures that different PCFS simulations are drawn for the combination
            i2 = np.random.randint(0, g2s.__len__())
        new_g2s.append((g2s[i1]+g2s[i2])/2)
        new_params.append((params[i2]+params[i2])/2)

    g2s = g2s + new_g2s
    params = params + new_params
    return g2s, params


if __name__ == '__main__':
    from g2_pcfs.modules.SimulatedPCFS import generate_pcfs
    from g2_pcfs.visualization.AP_figs_funcs import *
    from matplotlib import pyplot as plt

    bounds_path = '../g2-pcfs/data/data_grid_params.xlsx'

    n = 16

    param_permutations = param_permutations_random(bounds_path, n)

    g2s, params, spectra, t, df, nstage = make_training_g2s_2d(
        param_permutations,
        generate_pcfs,
        randomize_lineshape=0.5,
        randomize_diffusion=0.5,
    )

    g2s, params = augment_training_g2s_2d(g2s, params, int(n/2))

    make_fig((3, 3))
    for g2 in g2s:
        plt.plot(np.exp(t), g2.T, color='k', alpha=0.05)

    dress_fig(xlabel='Time (ps)', ylabel='$g^{(2)}(τ)$')
    plt.xscale('log')


    # idxs = [0, 1, 2, 10, 20, 40, 60]
    # fig, ax = plt.subplots(nrows=4, ncols=4, dpi=150, figsize=(8, 4))
    # ax = ax.flatten()
    # for a in range(0, 16):
    #     g2_sample = g2s[a]
    #
    #     ax[a].imshow(g2_sample,
    #                  vmin=0.5, vmax=1.05
    #                  )


        # for i in idxs:
        #     ax[a].plot(t, g2_sample[i, :], label=i)
        #     ax[a].set_ylim([0.5, 1.1])
    # plt.legend()
    # plt.tight_layout()



    # rand_inds = np.random.randint(low=0, high=len(g2s[:, 0]), size=50)
    # plt.figure()
    # plt.ylim([0.5, 1.05])
    # for i in rand_inds:
    #     plt.plot(t, g2s[i, :], label=i)
    # # plt.legend()





    # params_random[:, 13]
    # params_random[:, 14] = params_random[:, 14]/p_divider
    # params_random[:, 15] = params_random[:, 15]/p_divider

    #
    # idx = rand_inds[0]
    # idx = 22
    # print('α: %.6f' % params[14, idx])
    # # print('r: %.6f' % params[15, idx])
    # print('p: %.6f' % params[13, idx])
    # print('prod: %.6f' % (params[13, idx]*params[14, idx]))
    #

    """ Saving dataset """
    # basepath = "../data/raw/"
    # filepath = 'pcfs_g2_2d_n9999.h5'
    # import h5py
    # with h5py.File(basepath+filepath, "a") as h5_data:
    #     h5_data["g2s"] = g2s
    #     h5_data["params"] = params
    #     h5_data["t"] = t
    #     h5_data["df"] = df
    #     h5_data["nstage"] = nstage