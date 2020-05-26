#
import pickle
import matplotlib.pyplot as plt
import numpy as np
import details_plots as dp

file_high = "/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/papern_bradshaw_simulation_high_fn/disk/sim0/171/six_euv/display_obs_and_model.observations_vs_model_fits.171.six_euv.y=105,x=107.pkl"

file_intermediate = "/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/papern_bradshaw_simulation_intermediate_fn/disk/sim0/171/six_euv/display_obs_and_model.observations_vs_model_fits.171.six_euv.y=105,x=107.pkl"

labels = []

plt.ion()
plt.close('all')

fig, ax = plt.subplots()
xformatter = plt.FuncFormatter(dp.log_10_product)
ax.set_xscale('log')
ax.xaxis.set_major_formatter(xformatter)
ax.set_yscale('log')

for sim_type in ('high', 'intermediate'):
    if sim_type == 'high':
        final_filepath = file_high
    else:
        final_filepath = file_intermediate

    file_out = open(final_filepath, 'rb')
    study_type = pickle.load(file_out)
    best_fits = pickle.load(file_out)
    f = pickle.load(file_out)
    this_pwr = pickle.load(file_out)
    fn = pickle.load(file_out)
    fit_parameters = pickle.load(file_out)
    spectral_components = pickle.load(file_out)
    mc = pickle.load(file_out)
    file_out.close()

    bfs = mc.best_fit()
    ny = bfs.shape[0]
    nx = bfs.shape[1]
    percentile = [68.0, 32.0]
    bfup = np.percentile(bfs, percentile[1], axis=(0, 1))
    bflo = np.percentile(bfs, percentile[0], axis=(0, 1))

    linewidth = 1
    linestyle = "-"

    if sim_type == 'high':
        color = 'r'
        pwi = "$n=2.43\pm0.13$"
    else:
        color = 'black'
        pwi = "$n=2.36\pm0.16$"

    labels.append('{:s} power spectrum'.format(sim_type))
    ax.plot(f, this_pwr, label=labels[-1],
               linewidth=linewidth, color=color, linestyle=linestyle)
    labels.append('{:s} best fit {:s}'.format(sim_type, pwi))
    ax.plot(f, best_fits, label=labels[-1],
               linewidth=3*linewidth, color=color, linestyle="--")
    ax.set_xlabel(r'frequency $\nu$ (mHz)', fontsize=dp.fontsize)
    ax.set_ylabel('power (arbitrary units)', fontsize=dp.fontsize)
    ax.set_title('simulated spectrum comparison\n at pixel (105, 107)', fontsize=dp.fontsize)
    #plt.loglog(f, bfup, label='best fit ({:s})'.format(sim_type))
    #plt.loglog(f, bflo, label='best fit ({:s})'.format(sim_type))

#ax.axvline(1000.0/300.0, linestyle='--', color='blue',
#           linewidth=3*linewidth, label='5 minutes')
#ax.axvline(1000.0/180.0, linestyle='-.', color='blue',
#           linewidth=3*linewidth, label='3 minutes')
ax.set_ylim(10.**-9, 100.0)

legend = ax.legend(loc='lower left', framealpha=0.9, fontsize=0.9*dp.fontsize)
frame = legend.get_frame()
frame.set_facecolor('yellow')

fig.tight_layout()
plt.tick_params(axis='both', labelsize=0.8*dp.fontsize)
plt.savefig('/home/ireland/Desktop/spectrum_comparison.png', bbox_inches='tight')
