import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
import matplotlib.cm as cm
from tools.statistics import SummaryStatistics
import details_study as ds
from masking import VariableBounds, Fitness, IntensityMask

rc('text', usetex=True)  # Use LaTeX

# Which model to look at
observation_model_name = 'pl_c'
window = 'hanning'
power_type = 'absolute'

# Number of equally spaced bins in the histogram
bins = 50

# Colour for excluded fits in the spatial distribution
excluded_color = 'black'

waves = ['171']


for wave in waves:
    # General notification that we have a new data-set
    print('\nLoading New Data')

    # branch location
    b = [ds.corename, ds.original_datatype, wave]

    # Region identifier name
    region_id = ds.datalocationtools.ident_creator(b)

    # Location of the project data
    directory = ds.datalocationtools.save_location_calculator(ds.roots, b)["project_data"]

    # Load in some information about how to treat and plot the outputs, for example
    # output_name,lower_bound,upper_bound,variable_name
    # "amplitude_0",None,None,"A_{0}"
    # "alpha_0",0,4,"n"
    filename = 'models.outputs_information.{:s}.csv'.format(observation_model_name)
    # filepath = os.path.join(os.path.expanduser('~/time_series_cor_heat-git/time_series_cor_heat'), filename)
    df = pd.read_csv(filename, index_col=0)
    df = df.replace({"None": None})

    # Load in the fit parameters and the output names
    filename = '{:s}_{:s}_{:s}_{:s}.{:s}.outputs.step3.npz'.format(observation_model_name,
                                                                   ds.study_type, wave, window,
                                                                   power_type)
    filepath = os.path.join(directory, filename)
    print(f'Loading {filepath}')
    outputs = np.load(filepath)['arr_0']

    filename = '{:s}_{:s}_{:s}_{:s}.{:s}.names.step3.txt'.format(observation_model_name,
                                                                   ds.study_type, wave, window,
                                                                   power_type)
    filepath = os.path.join(directory, filename)
    print(f'Loading {filepath}')
    with open(filepath) as f:
        output_names = [line.rstrip() for line in f]

    # Load in the fits
    filename = '{:s}_{:s}_{:s}_{:s}.{:s}.mfits.step3.npz'.format(observation_model_name,
                                                                   ds.study_type, wave, window,
                                                                   power_type)
    filepath = os.path.join(directory, filename)
    print(f'Loading {filepath}')
    mfits = np.load(filepath)['arr_0']
    freq = np.load(filepath)['arr_1']

    # Load in the analysis details
    filename = '{:s}_{:s}_{:s}_{:s}.{:s}.analysis.step3.npz'.format(observation_model_name,
                                                                   ds.study_type, wave, window,
                                                                   power_type)
    filepath = os.path.join(directory, filename)
    print(f'Loading {filepath}')
    subsection = np.load(filepath)['arr_0']
    normalize_frequencies = np.all(np.load(filepath)['arr_1'])
    divide_by_initial_power = np.all(np.load(filepath)['arr_2'])

    # Load in the observed fourier power data
    filename = '{:s}_{:s}_{:s}.{:s}.step2.npz'.format(ds.study_type, wave, window, power_type)
    filepath = os.path.join(directory, filename)
    print(f'Loading {filepath}')
    observed = (np.load(filepath)['arr_0'])[subsection[0]:subsection[1], subsection[2]:subsection[3], :]
    if divide_by_initial_power:
        for i in range(0, observed.shape[0]):
            for j in range(0, observed.shape[1]):
                observed[i, j, :] = observed[i, j, :] / observed[i, j, 0]

    # Load in the original time series data to create an intensity mask
    filename = '{:s}_{:s}.step1.npz'.format(ds.study_type, wave)
    filepath = os.path.join(directory, filename)
    print(f'Loading {filepath}')
    emission = (np.load(filepath)['arr_0'])[subsection[0]:subsection[1], subsection[2]:subsection[3], :]

    # Calculate a mask.  The mask eliminates results that we do not wish to consider,
    # for example, excluded fits.  The mask is calculated using all the variable
    # output.  True values will be masked out
    mask = np.zeros_like(outputs[:, :, 0], dtype=bool)
    shape = mask.shape
    ny = shape[0]
    nx = shape[1]
    for i, output_name in enumerate(output_names):
        data = outputs[:, :, i]

        # Finiteness
        is_not_finite = ~np.isfinite(data)
        mask = np.logical_or(mask, is_not_finite)

        # Boundaries
        bounds = (df['lower_bound'][output_name], df['upper_bound'][output_name])
        boundaries = VariableBounds(data, bounds)

        # Update mask
        mask = np.logical_or(mask, boundaries.exceeds_low)
        mask = np.logical_or(mask, boundaries.exceeds_high)

    # Calculate a fitness mask
    fitness_mask = np.zeros_like(mask)
    for i in range(0, nx):
        for j in range(0, ny):
            fitness = Fitness(observed[i, j, :], mfits[i, j, :], 3)
            fitness_mask[i, j] = not fitness.is_good()

    # Update the overall mask with the fitness mask
    mask = np.logical_or(mask, fitness_mask)

    # Calculate a brightness mask
    total_intensity = np.sum(emission, axis=2)
    intensity_mask = IntensityMask(total_intensity, 0.01)

    # Update the overall mask with the fitness mask
    mask = np.logical_or(mask, intensity_mask)

    # Make the plots
    super_title = "{:s}, {:s}\n".format(ds.study_type.replace("_", " "), wave)
    for i, output_name in enumerate(output_names):
        # Transpose because the data is the wrong way around
        data = np.transpose(np.ma.array(outputs[:, :, i], mask=mask))

        # Total number of fits, including excluded ones
        n_samples = data.size

        # Compressed data and the number of good and excluded fits
        compressed = data.flatten().compressed()
        n_good = compressed.size
        n_excluded = n_samples - n_good

        # Summary statistics
        ss = SummaryStatistics(compressed, ci=(0.16, 0.84, 0.025, 0.975), bins=bins)

        # The variable name is used in the plot instead of the output_name
        # because we use LaTeX in the plots to match with the variables
        # used in the paper.
        variable_name = df['variable_name'][output_name]

        # Percentage that are excluded fits
        percent_excluded_string = "{:.1f}$\%$".format(100*n_excluded/n_samples)

        # Information that goes in to the histogram title
        title_information = f"{variable_name}\n{n_samples} fits, {n_excluded} excluded, {n_good} good, {percent_excluded_string} excluded"

        # Credible interval strings
        ci_a = "{:.1f}$\%$".format(100*ss.ci[0])
        ci_b = "{:.1f}$\%$".format(100*ss.ci[1])
        ci_c = "{:.1f}$\%$".format(100*ss.ci[2])
        ci_d = "{:.1f}$\%$".format(100*ss.ci[3])
        ci_1 = 'C.I. {:s}$\\rightarrow${:s} ({:.2f}$\\rightarrow${:.2f})'.format(ci_a, ci_b, ss.cred[0], ss.cred[1])
        ci_2 = 'C.I. {:s}$\\rightarrow${:s} ({:.2f}$\\rightarrow${:.2f})'.format(ci_c, ci_d, ss.cred[2], ss.cred[3])

        # Histograms
        plt.close('all')
        fig, ax = plt.subplots()
        h = ax.hist(compressed, bins=bins)
        plt.xlabel(variable_name)
        plt.ylabel('number')
        plt.title(f'{super_title}histogram of {title_information}')
        plt.grid(linestyle=":")
        ax.axvline(ss.mean, label='mean ({:.2f})'.format(ss.mean), color='r')
        ax.axvline(ss.mode, label='mode ({:.2f})'.format(ss.mode), color='k')
        ax.axvline(ss.median, label='median ({:.2f})'.format(ss.median), color='y')
        ax.axvline(ss.cred[0], color='r', linestyle=':')
        ax.axvline(ss.cred[1], label=ci_1, color='r', linestyle=':')
        ax.axvline(ss.cred[2], color='k', linestyle=':')
        ax.axvline(ss.cred[3], label=ci_2, color='k', linestyle=':')
        ax.legend()
        filename = 'histogram.{:s}.{:s}.png'.format(observation_model_name, output_name)
        filename = os.path.join(directory, filename)
        plt.savefig(filename)

        # Spatial distribution
        title_information = f"{variable_name}\n{n_samples} fits, {n_excluded} excluded (in {excluded_color}), {n_good} good, {percent_excluded_string} excluded"
        plt.close('all')
        fig, ax = plt.subplots()
        if output_name == 'alpha_0':
            cmap = cm.Dark2_r
            im = ax.imshow(data, origin='lower', cmap=cmap)
            im.set_clim(df['lower_bound'][output_name], df['upper_bound'][output_name])
        else:
            cmap = cm.inferno
            im = ax.imshow(data, origin='lower', cmap=cmap)
        im.cmap.set_bad(excluded_color)
        ax.set_xlabel('solar X')
        ax.set_ylabel('solar Y')
        ax.set_title(f'{super_title}spatial distribution of {title_information}')
        ax.grid(linestyle=":")
        fig.colorbar(im, ax=ax, label=variable_name)
        filename = 'spatial.{:s}.{:s}.png'.format(observation_model_name, output_name)
        filename = os.path.join(directory, filename)
        plt.savefig(filename)

    # Plot some example spectra
    nx_plot = 3
    ny_plot = 3
    nx = mfits.shape[0]
    ny = mfits.shape[1]
    fig, axs = plt.subplots(nx_plot, ny_plot)
    fig.figsize = (2*nx_plot, 2*ny_plot)
    for i in range(0, nx_plot):
        for j in range(0, ny_plot):
            ii = np.random.randint(0, nx)
            jj = np.random.randint(0, ny)
            while mask[ii, jj]:
                ii = np.random.randint(0, nx)
                jj = np.random.randint(0, ny)
            axs[i, j].loglog(freq, observed[ii, jj, :])
            axs[i, j].loglog(freq, mfits[ii, jj, :])
            axs[i, j].set_title('{:n},{:n}'.format(ii, jj))
            axs[i, j].grid('on', linestyle=':')

    fig.tight_layout()
    filename = 'sample_fits.{:s}.{:s}.png'.format(observation_model_name, output_name)
    filename = os.path.join(directory, filename)
    plt.savefig(filename)
