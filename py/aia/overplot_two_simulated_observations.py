#
import pickle
import matplotlib.pyplot as plt
import numpy as np

file_high = "/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/papern_bradshaw_simulation_high_fn/disk/sim0/171/six_euv/display_obs_and_model.observations_vs_model_fits.171.six_euv.y=105,x=107.pkl"

file_intermediate = "/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/papern_bradshaw_simulation_intermediate_fn/disk/sim0/171/six_euv/display_obs_and_model.observations_vs_model_fits.171.six_euv.y=105,x=107.pkl"

plt.close('all')
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

    plt.loglog(f, this_pwr, label='power spectrum ({:s})'.format(sim_type))
    plt.loglog(f, best_fits, label='best fit ({:s})'.format(sim_type))


plt.show()

