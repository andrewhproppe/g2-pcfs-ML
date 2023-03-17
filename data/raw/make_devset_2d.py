import h5py

from g2_pcfs.pipeline.make_dataset import param_permutations_random, make_training_g2s_2d, augment_training_g2s_2d
from g2_pcfs.modules.SimulatedPCFS import generate_pcfs

# bounds_path = (f"../data/data_grid_params.xlsx")
bounds_path = ("data/data_grid_params.xlsx")

# number of g2s to generate and number to augment
n = 30
naug = int(n/3)

# Randomly samples parameters for generating simulated PCFS objects. Each object contains 30 g2s
param_permutations = param_permutations_random(bounds_path, n)

# Make the g2s
g2s, params, spectra, t, df, nstage = make_training_g2s_2d(
    param_permutations,
    generate_pcfs,
    0.5,
    0.5,
)

# Create linear combinations of the g2s to augment training data
g2s, params = augment_training_g2s_2d(
    g2s,
    params,
    naug,
)

# # Plot to verify the variance of the g2s
# from matplotlib import pyplot as plt
# for g2 in g2s:
#     plt.plot(t, g2, color='k', alpha=0.1)
#
# Save the data to .h5 file
basepath = "data/raw/"
filepath = 'pcfs_g2_2d_n%i.h5' % (n+naug)

with h5py.File(basepath+filepath, "a") as h5_data:
    h5_data["g2s"] = g2s
    h5_data["params"] = params
    h5_data["t"] = t
    h5_data["df"] = df
    h5_data["nstage"] = nstage