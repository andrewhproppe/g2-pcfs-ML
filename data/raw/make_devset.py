import h5py

from g2_pcfs.pipeline.make_dataset import param_permutations_random, make_training_g2s, augment_training_g2s
from g2_pcfs.modules.SimulatedPCFS import generate_pcfs

bounds_path = '../data_grid_params.xlsx'

# number of g2s to generate and number to augment
n = 10
naug = n/3
randomize_lineshape = 0.5
randomize_diffusion = 0.5

# Randomly samples parameters for generating simulated PCFS objects. Each object contains 30 g2s
param_permutations = param_permutations_random(bounds_path, n)

# Make the g2s
g2s, params, t, df, nstage = make_training_g2s(
    param_permutations,
    generate_pcfs,
    randomize_lineshape,
    randomize_diffusion
)

naug = int(naug*nstage)

# Create linear combinations of the g2s to augment training data
g2s, params = augment_training_g2s(
    g2s,
    params,
    nstage,
    naug,
)

# # Plot to verify the variance of the g2s
# from matplotlib import pyplot as plt
# for g2 in g2s:
#     plt.plot(t, g2, color='k', alpha=0.1)

#
# # Save the data to .h5 file
# basepath = ""
# filepath = 'pcfs_g2_n%i.h5' % (g2s.shape[0])
#
# with h5py.File(basepath+filepath, "a") as h5_data:
#     h5_data["g2s"] = g2s
#     h5_data["params"] = params
#     h5_data["t"] = t
#     h5_data["df"] = df
#     h5_data["nstage"] = nstage
#     h5_data["randomize_lineshape"] = randomize_lineshape
#     h5_data["randomize_diffusion"] = randomize_diffusion