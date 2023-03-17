import pickle
import torch
from g2_pcfs.visualization.AP_figs_funcs import *
from g2_pcfs import predict
from g2_pcfs.visualization.visualize import plot_ensemble_variance, plot_ensemble_submodels
from g2_pcfs.pipeline.data import g2DataModule
if not matplotlib.get_backend() == 'MacOSX':
    matplotlib.use('MacOSX')

dot_path = '../../dots/'
# dot_name = 'CK174_dot6_full' # GOOD even with g2
# dot_name = 'CK174_dot6_updated.pickle'
# dot_name = 'dot4_zwitTPPO_2uW_20210819.pickle'
# dot_name = 'dotC_8uW_20210523.pickle'
# dot_name = 'dotC_20210604.pickle'
# dot_name = 'dotH_8uW_20210524.pickle' # GOOD even with g2
# dot_name = 'dotL_1uW_20210604.pickle'
dot_name = 'dot2_overnight.pickle'

# pickle_in = open(dot_path+dot_name, "rb"); dot = pickle.load(pickle_in); pickle_in.close()

import seaborn as sns

def plot_ensemble_variance_dot(dot, nstd=2, idx=0, cidxs=np.array([3, 8, 1, 10]), figsize=(4, 4), colors=None, lgnd_cols=2, lgd_loc=None, fixframe=False):
    if fixframe:
        fig, ax = make_fig(figsize)
    else:
        fig, ax = plt.figure(figsize)

    if colors is None:
        colors = np.array(sns.color_palette("icefire", 12))

    τ = dot.tau_interp
    μ = dot.g2s_pred_μ[idx, :]
    σ = dot.g2s_pred_σ[idx, :]
    plt.plot(dot.tau, dot.g2[idx, :], label='Input', color=colors[cidxs[0]])
    plt.plot(τ, dot.g2s_interp[idx, :], label='Interp.', color=colors[cidxs[2]])
    plt.plot(τ, μ, label='$\mu$', color='k', zorder=10)
    fill_col = [0, 0, 0]
    plt.fill_between(τ, μ + nstd * σ, μ - nstd * σ, linewidth=0.0, color=fill_col, alpha=0.3, label='$\mu$ +/- %d$\sigma$' % nstd, zorder=5)
    dress_fig(xlim=[1e5, 1e11], ylim=[0.5, 1.1], xlabel='$\\tau$ ($\mu$s)', ylabel='$g^{(2)}(\\tau)$', legend=False, tight=False, lgnd_cols=lgnd_cols, lgd_loc=None)
    plt.xscale('log')

    return fig

# 3, 11x, 225, 302, 303, 304, 306, 307, 308 200, 300
# 3, 225, 302, 304
sim_idx = 8010
data = g2DataModule('pcfs_g2_n25000.h5', batch_size=512, window_size=1)
data.setup()
(x, y, params) = data.train_set.__getitem__(sim_idx)


model_name = 'AdvConv1DEnsemble'
ckpt = 'Epoch199.ckpt'
model = predict.get_checkpoint(model_name, ckpt)
X = torch.Tensor(x).unsqueeze(0)
with torch.no_grad():
    pred = model(X)
    g2s_pred_μ = pred[0].detach().numpy()
    g2s_pred_v = pred[1].detach().numpy()
    g2s_pred_σ = np.sqrt(g2s_pred_v)
    g2s_pred_submodels = pred[2].detach().numpy()
    g2_AAE = tuple((g2s_pred_μ, g2s_pred_σ, g2s_pred_submodels))


def plot_ensemble_variance_sim(g2_AAE, nstd=2, cidxs=np.array([3, 8, 1, 10]), figsize=(4, 4), colors=None, lgnd_loc=None, lgnd_cols=2, fixframe=False):
    if fixframe:
        fig, ax = make_fig(figsize)
    else:
        fig, ax = plt.figure(figsize)

    if colors is None:
        colors = np.array(sns.color_palette("icefire", 12))

    g2s_pred_μ = g2_AAE[0]
    g2s_pred_σ = g2_AAE[1]
    τ = np.exp(x[:, 0])
    μ = g2s_pred_μ[0, :]
    σ = g2s_pred_σ[0, :]
    plt.plot(τ, x[:, 1], label='Input', color=colors[cidxs[0]])
    plt.plot(τ, y[:, 1], label='True', color=colors[cidxs[-1]])
    plt.plot(τ, μ, label='$\mu$', color='k')
    fill_col = [0, 0, 0]
    plt.fill_between(τ, μ+nstd*σ, μ-nstd*σ, linewidth=0.0, color=fill_col, alpha=0.3, label='$\mu$ +/- %d$\sigma$' % nstd)
    dress_fig(xlim=[1e5, 1e11], ylim=[-0.05, 1.05], xlabel='$\\tau$ (ps)', ylabel='$g^{(2)}(\\tau)$', tight=False,
              legend=False, lgnd_cols=lgnd_cols, lgd_loc=lgnd_loc
              )
    # plt.xscale('log')

    return fig

set_font_size(7, lgnd=0)
figsize = (1.5, 1.)

plot_ensemble_variance_sim(
    g2_AAE,
    nstd=1,
    fixframe=True,
    figsize=figsize
)

# plot_ensemble_variance_dot(
#     dot=dot,
#     fixframe=True,
#     figsize=figsize,
#     nstd=1,
#     idx=70
# )
plt.xscale('log')


# plot_ensemble_submodels(
#     x=np.exp(x[:, 0]),
#     g2_ens=dot.g2_AEE,
#     figsize=(3, 2),
#     spacer=0.0,
#     idx=40
# )
plt.xscale('log')

