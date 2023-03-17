import pickle
import sys
import matplotlib.pyplot as plt
import torch
from g2_pcfs.visualization.AP_figs_funcs import *
from g2_pcfs import predict
if not matplotlib.get_backend() == 'MacOSX':
    matplotlib.use('MacOSX')
sys.path.append('../g2_pcfs/modules')

""" Opens a dot.pickle object and grabs the cross correlation data from it. The cross correlations need to undergo a series of processing steps in 
 prepare_dot_g2s() before being prepared into a batch tensor that is fed into the ensemble model that returns the prediction for all g2s in a single pass. 
 The mean (μ), variance (var), standard deviation (σ), and submodels are all calculated and then attached as attributes to the dot objected, which is then 
 repickled and can be used for processing in all other PCFS scripts."""

dot_path = '../dots/'
dot_name = 'CK174_dot6_full' # GOOD even with g2
# dot_name = 'CK174_dot6_updated.pickle'
# dot_name = 'dot4_zwitTPPO_2uW_20210819.pickle'
# dot_name = 'dotC_8uW_20210523.pickle'
# dot_name = 'dotC_20210604.pickle'
# dot_name = 'dotH_8uW_20210524.pickle' # GOOD even with g2
# dot_name = 'dotL_1uW_20210604.pickle'
# dot_name = 'dot2_overnight.pickle'

pickle_in = open(dot_path+dot_name, "rb"); dot = pickle.load(pickle_in); pickle_in.close()

# raise RuntimeError

def prepare_dot_g2s(dot, input_dim=140, min_τ=1e4, max_τ=1e11):
    """ Takes in a dot object used for PCFS experiments with N g2 functions (one per stage position) and T time bins
    The neural networks models have a fixed input size of input_dim=140 time bins and were trained on g2 functions with photons correlated between
    τ = 1e4 - 1e11 ps. This function first takes dot.tau and takes it log before defining a truncating sparser array τ that matches input_dim using linspace

    g2s functions are then taken from the dot, spurious negative values deleted, and each g2 is normalized to 1. These maximum values are stored in an array,
    so that they can be used to restore the raw g2 and the model prediction to the original scale. The final shape of the g2s_norm array is [N, T, 2], where τ
    is stored along with the g2 and fed into the network (the time array doesn't actually play a role in the nn but was including in case time derivatives would
     ever become useful in the forward step of different networks)."""
    t_max_ind     = find_nearest(max_τ, dot.tau)
    t_min_ind     = find_nearest(min_τ, dot.tau)
    log_dot_τ     = np.log(dot.tau)[0:t_max_ind] # need to take log of dot time so we can use linspace to interpolate later to match nn input. Truncate whatever the dots max time bound is to 1e11; this matches the nn training data
    τ             = np.linspace(log_dot_τ[0], log_dot_τ[-1], input_dim) # sets the number of time bins to 140; matches nn input dimension

    g2s = dot.g2
    g2s[g2s < 0] = 0
    g2s = g2s[:, t_min_ind:t_max_ind]
    g2_max_vals = np.zeros(g2s.shape[0])
    g2s_norm = np.zeros((g2s.shape[0], input_dim, 2))
    for i, g2 in enumerate(g2s):
        g2 = np.interp(τ, log_dot_τ, g2)
        g2_max_vals[i] = max(g2)
        g2s_norm[i, :, 0] = τ
        g2s_norm[i, :, 1] = g2/max(g2)

    X = torch.Tensor(g2s_norm)
    return X, τ, g2_max_vals, g2s_norm

X, τ, g2_max_vals, g2s_norm = prepare_dot_g2s(dot)

""" Prediction with AdvConv1DEnsemble """
model_name = 'AdvConv1DEnsemble'
ckpt = 'Epoch199.ckpt'
model = predict.get_checkpoint(model_name, ckpt)

""" Get predictions from AAE model """
with torch.no_grad():
    pred = model(X)
    g2s_pred_μ = pred[0].detach().numpy()
    g2s_pred_v = pred[1].detach().numpy()
    g2s_pred_σ = np.sqrt(g2s_pred_v)
    g2s_pred_submodels = pred[2].detach().numpy()
    g2_AAE = tuple((g2s_pred_μ, g2s_pred_σ, g2s_pred_submodels))

""" Prediction with multiple single models """
def multi_model_g2_prediction(model_name, ckpts, X):
    pred_arr = torch.zeros(len(ckpts), X.shape[0], X.shape[1])
    for i, ckpt in enumerate(ckpts):
        model = predict.get_checkpoint(model_name, ckpt)
        model.eval()
        with torch.no_grad():
            temp_pred = model(X)
            pred_arr[i, :, :] = temp_pred[0]

    g2s_pred_submodels = pred_arr.detach().numpy()
    g2s_pred_μ = g2s_pred_submodels.mean(axis=0)
    g2s_pred_var = g2s_pred_submodels.var(axis=0)
    g2s_pred_σ = np.sqrt(g2s_pred_var)
    g2_AAE = tuple((g2s_pred_μ, g2s_pred_σ, g2s_pred_submodels))
    return g2s_pred_μ, g2s_pred_σ, g2s_pred_submodels, g2_AAE


""" Get predictions from multiple single Conv1DAutoEncoder models """
# model_name = 'Conv1DAutoEncoder'
# ckpts = [
#     "Conv1DAutoEncoder_kernel3_model1.ckpt",
#     "Conv1DAutoEncoder_kernel3_model2.ckpt",
#     "Conv1DAutoEncoder_kernel3_model3.ckpt",
#     "Conv1DAutoEncoder_kernel3_model4.ckpt",
#     "Conv1DAutoEncoder_kernel3_model5.ckpt",
#     # "Conv1DAutoEncoder_kernel7.ckpt",
#     # "Conv1DAutoEncoder_kernel11.ckpt",
#     # "Conv1DAutoEncoder_kernel15.ckpt",
# ]
# g2s_pred_μ, g2s_pred_σ, g2s_pred_submodels, g2_AAE = multi_model_g2_prediction(model_name, ckpts, X)

""" Remove tau remove raw g2 array and rescale g2s back to original intensities (un-normalize) """
g2_intensites      = g2s_norm[:, :, 1] # Get intensites back out of g2
g2s_norm           = g2_intensites # Get rid of τ from g2_norm array, now that it isn't needed after running through nn model
g2s_interp         = g2s_norm * g2_max_vals[:, None]
g2s_pred_μ         = g2s_pred_μ * g2_max_vals[:, None]
g2s_pred_σ         = g2s_pred_σ * g2_max_vals[:, None]
g2s_pred_submodels = g2s_pred_submodels * g2_max_vals[:, None]


""" At this point, all predictions and scaling are finished. All that is left is to store the arrays into back into the dot object for PCFS plotting """
setattr(dot, "tau_interp", np.exp(τ)) # return to linear scale before storing
setattr(dot, "g2s_interp", g2s_interp)
setattr(dot, "g2s_pred_μ", g2s_pred_μ)
setattr(dot, "g2s_pred_σ", g2s_pred_σ)
setattr(dot, "g2s_pred_submodels", g2s_pred_submodels)
setattr(dot, "g2_AEE", g2_AAE)
pickle_out = open(dot_path+dot_name, "wb"); pickle.dump(dot, pickle_out); pickle_out.close()  # save pickle file after correlations are performed, to avoid performing them again when reloading


# Some plotting for verification

τ = np.exp(τ)
make_fig((3, 2))
idx = -1
# plt.plot(dot.g2_x[0, :])
# plt.plot(dot.g2[idx, :])
# plt.plot(dot.tau, dot.g2[idx, :])
plt.plot(τ, g2s_interp[idx, :], 'k')
plt.plot(τ, g2s_pred_μ[idx, :])
# plt.plot(τ, g2s_pred_μ[idx, :] + 2*g2s_pred_σ[idx, :])
# plt.plot(τ, g2s_pred_μ[idx, :] - 2*g2s_pred_σ[idx, :])
# plt.plot(τ, g2s_pred_submodels[0, 0, :140])
# plt.ylim([0.75, 1.02])
plt.xscale('log')