import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from matplotlib import cm
import sunpy.visualization.colormaps as scm
import matplotlib.colors as colors
from astropy.visualization import AsinhStretch, ImageNormalize
from tools.statistics import SummaryStatistics
import details_study as ds

parser = argparse.ArgumentParser(description='Plot maps and histograms of results')
parser.add_argument('-w', '--waves', help='comma separated list of channels', type=str)
parser.add_argument('-s', '--study', help='comma separated list of study types', type=str)
parser.add_argument('-p', '--plots', help='comma separated list of plot types ("individual", "gang_by_wave", "gang_by_simulation_and_wave"', type=str)
args = parser.parse_args()
waves = [item for item in args.waves.split(',')]
plots = [item for item in args.plots.split(',')]

# Studies to load in
study_types = [item for item in args.study.split(',')]
study_type = study_types[0]

rc('text', usetex=True)  # Use LaTeX

# Which model to look at
observation_model_name = 'pl_c'
window = 'hanning'
power_type = 'absolute'

# Number of equally spaced bins in the histogram
bins = 50

# Colour for excluded fits in the spatial distribution
excluded_color = 'black'


def plot_histogram(ax, compressed, bins, variable_name, title, show_statistics=True):
    """
    Creates a histogram plot.

    Parameters
    ----------
    ax:
    compressed:
    bins:
    variable_name:
    title:

    Returns
    -------
    """

    # Create and return the plot
    h = ax.hist(compressed, bins=bins)
    ax.set_xlabel(variable_name)
    ax.set_ylabel('number')
    ax.set_title(title)
    ax.grid(linestyle=":")

    if show_statistics:
        # Get the summary statistics
        ss = SummaryStatistics(compressed, ci=(0.16, 0.84, 0.025, 0.975), bins=bins)
        # Credible interval strings for the plot
        ci_a = "{:.1f}$\%$".format(100 * ss.ci[0])
        ci_b = "{:.1f}$\%$".format(100 * ss.ci[1])
        ci_c = "{:.1f}$\%$".format(100 * ss.ci[2])
        ci_d = "{:.1f}$\%$".format(100 * ss.ci[3])
        ci_1 = 'C.I. {:s}$\\rightarrow${:s} ({:.2f}$\\rightarrow${:.2f})'.format(ci_a, ci_b,
                                                                                 ss.cred[0],
                                                                                 ss.cred[1])
        ci_2 = 'C.I. {:s}$\\rightarrow${:s} ({:.2f}$\\rightarrow${:.2f})'.format(ci_c, ci_d,
                                                                                 ss.cred[2],
                                                                                 ss.cred[3])
        ax.axvline(ss.mean, label='mean ({:.2f})'.format(ss.mean), color='r')
        ax.axvline(ss.mode, label='mode ({:.2f})'.format(ss.mode), color='k')
        ax.axvline(ss.median, label='median ({:.2f})'.format(ss.median), color='y')
        ax.axvline(ss.cred[0], color='r', linestyle=':')
        ax.axvline(ss.cred[1], label=ci_1, color='r', linestyle=':')
        ax.axvline(ss.cred[2], color='k', linestyle=':')
        ax.axvline(ss.cred[3], label=ci_2, color='k', linestyle=':')
    ax.legend()
    return ax


def plot_spatial_distribution(ax, data, output_name, title):
    """
    Creates a spatial distribution plot.

    Parameters
    ----------
    ax:
    data:
    output_name:
    title:

    Returns
    -------
    """

    if output_name == 'alpha_0':
        cmap = cm.Dark2_r
        im = ax.imshow(data, origin='lower', cmap=cmap,
                       norm=colors.Normalize(vmin=0.0, vmax=4.0, clip=False))
        im.cmap.set_over('lemonchiffon')
    elif "err_" in output_name:
        cmap = cm.plasma
        im = ax.imshow(data, origin='lower', cmap=cmap,
                       norm=colors.LogNorm(vmin=compressed.min(), vmax=compressed.max()))
    else:
        cmap = cm.plasma
        im = ax.imshow(data, origin='lower', cmap=cmap)
    im.cmap.set_bad(excluded_color)
    ax.set_xlabel('solar X')
    ax.set_ylabel('solar Y')
    ax.set_title(title)
    ax.grid(linestyle=":")
    return im, ax


def plot_emission(ax, data, title, intensity_cmap):
    """
    Creates an emission plot
    """
    norm = ImageNormalize(stretch=AsinhStretch(0.01), vmin=data.min(), vmax=data.max())
    im = ax.imshow(data, origin='lower', cmap=intensity_cmap, norm=norm)
    im.cmap.set_bad(excluded_color)
    ax.set_xlabel('solar X')
    ax.set_ylabel('solar Y')
    ax.set_title(title)
    ax.grid(linestyle=":")
    return im, ax


# overlay_image_plot
def plot_overlay_image(ax, image, title):
    ax.imshow(image, origin='lower')    
    ax.set_xlabel('solar X')
    ax.set_ylabel('solar Y')
    ax.set_title(title)
    ax.grid(linestyle=":")
    return ax


# Overlay histograms
def plot_overlay_histograms(ax, results, bins, colors, labels, study_types, variable_name, title, density=False):
    """

    ax:
    results:
    bins:
    colors:
    keys:
    :return:
    """

    # For each study
    for study_type in study_types:
        data = results[study_type]
        h = ax.hist(data, bins=bins, alpha=0.5, color=colors[study_type], label=labels[study_type], density=density)

    ax.set_xlabel(variable_name)
    if not density:
        ax.set_ylabel('number')
    else:
        ax.set_ylabel('probability density')
    ax.set_title(title)
    ax.grid(linestyle=":")
    ax.legend()
    return ax


# Helper function
def mask_plotting_information(m, excluded_color=None):
    n_samples = m.size
    n_excluded = np.sum(m)
    n_good = n_samples - n_excluded
    percent_excluded_string = "{:.1f}$\%$".format(100 * n_excluded / n_samples)
    if excluded_color is None:
        mask_info = f"{n_samples} pixels, {n_excluded} [{percent_excluded_string}] excluded, {n_good} included"
    else:
        mask_info = f"{n_samples} pixels, {n_excluded} [{percent_excluded_string}] excluded (in {excluded_color}), {n_good} included"
    return mask_info


def load_masks(directory, base_filename):
    """
    Loads in all the masks in a given directory and base filename
    :param directory:
    :param base_filename:
    :return:
    """
    mask_list = ("finiteness", "bounds", "fitness", "intensity", "combined")
    masks = dict()
    for tm in mask_list:
        filename = f'{base_filename}.{tm}.step4.npz'
        filepath = os.path.join(directory, filename)
        print(f'Loading {filepath}')
        masks[tm] = (np.load(filepath))['arr_0']
    masks['none'] = np.zeros_like(masks['combined'])
    return masks


# Hack to get the output file names - all wavelengths are the same anyway.
def get_output_names_hack(study_type, ds, observation_model_name, window, power_type, wave='335'):
    b = [study_type, ds.original_datatype, wave]
    directory = ds.datalocationtools.save_location_calculator(ds.roots, b)["project_data"]
    base_filename = f"{observation_model_name}_{study_type}_{wave}_{window}.{power_type}"
    filename = f'{base_filename}.names.step3.txt'
    filepath = os.path.join(directory, filename)
    print(f'Loading {filepath}')
    with open(filepath) as f:
        output_names = [line.rstrip() for line in f]
    return output_names


def load_fit_parameters(directory, base_filename):
    """
    Load in the fit data
    """
    filename = f'{base_filename}.outputs.step3.npz'
    filepath = os.path.join(directory, filename)
    print(f'Loading {filepath}')
    return np.load(filepath)['arr_0']


def load_fit_parameter_output_names(directory, base_filename):
    """
    Load in the fit parameter output names
    """
    filename = f'{base_filename}.names.step3.txt'
    filepath = os.path.join(directory, filename)
    print(f'Loading {filepath}')
    with open(filepath) as f:
        output_names = [line.rstrip() for line in f]
    return output_names


# Load in some information about how to treat and plot the outputs, for example
# output_name,lower_bound,upper_bound,variable_name
# "amplitude_0",None,None,"A_{0}"
# "alpha_0",0,4,"n"
filename = 'models.outputs_information.{:s}.csv'.format(observation_model_name)
df = pd.read_csv(filename, index_col=0)
df = df.replace({"None": None})


# Plots per single AIA channel and simulation
if 'individual' in plots:

    # Load the data
    for wave in waves:

        # Colour maps
        if "verify_fitting" not in study_type:
            intensity_cmap = plt.get_cmap(f'sdoaia{wave}')
        else:
            intensity_cmap = plt.get_cmap('gray')

        # General notification that we have a new data-set
        print('\nLoading New Data')

        # branch location
        b = [study_type, ds.original_datatype, wave]

        # Location of the project data
        directory = ds.datalocationtools.save_location_calculator(ds.roots, b)["project_data"]

        # Base filename
        base_filename = f"{observation_model_name}_{study_type}_{wave}_{window}.{power_type}"

        # Load in the fit parameters and the output names
        filename = f'{base_filename}.outputs.step3.npz'
        filepath = os.path.join(directory, filename)
        print(f'Loading {filepath}')
        outputs = np.load(filepath)['arr_0']

        filename = f'{base_filename}.names.step3.txt'
        filepath = os.path.join(directory, filename)
        print(f'Loading {filepath}')
        with open(filepath) as f:
            output_names = [line.rstrip() for line in f]

        # Load in the fits
        filename = f'{base_filename}.mfits.step3.npz'
        filepath = os.path.join(directory, filename)
        print(f'Loading {filepath}')
        mfits = np.load(filepath)['arr_0']
        freq = np.load(filepath)['arr_1']

        # Load in the analysis details
        filename = f'{base_filename}.analysis.step3.npz'
        filepath = os.path.join(directory, filename)
        print(f'Loading {filepath}')
        subsection = np.load(filepath)['arr_0']
        normalize_frequencies = np.all(np.load(filepath)['arr_1'])
        divide_by_initial_power = np.all(np.load(filepath)['arr_2'])

        # Load in the observed fourier power data
        filename = f'{study_type}_{wave}_{window}.{power_type}.step2.npz'
        filepath = os.path.join(directory, filename)
        print(f'Loading {filepath}')
        observed = (np.load(filepath)['arr_0'])[subsection[0]:subsection[1], subsection[2]:subsection[3], :]
        if divide_by_initial_power:
            for i in range(0, observed.shape[0]):
                for j in range(0, observed.shape[1]):
                    observed[i, j, :] = observed[i, j, :] / observed[i, j, 0]

        # Load in the original time series data
        if "verify_fitting" not in study_type:
            filename = f'{study_type}_{wave}.step1.npz'
            filepath = os.path.join(directory, filename)
            print(f'Loading {filepath}')
            emission = (np.load(filepath)['arr_0'])[subsection[0]:subsection[1], subsection[2]:subsection[3], :]
        else:
            emission = np.ones_like(observed)

        # Load in the mask data
        mask_list = ("finiteness", "bounds", "fitness", "intensity", "combined")
        masks = dict()
        for this_mask in mask_list:
            filename = f'{base_filename}.{this_mask}.step4.npz'
            filepath = os.path.join(directory, filename)
            print(f'Loading {filepath}')
            masks[this_mask] = (np.load(filepath))['arr_0']

        # Calculate a brightness mask
        total_intensity = np.transpose(np.sum(emission, axis=2))

        ###########################
        # Make the plots
        # The super title describes the study type and the wavelength
        super_title = "{:s}, {:s}\n".format(study_type.replace("_", " "), wave)

        ###########################
        # Plot the masks
        for this_mask in mask_list:
            description = f'{this_mask} mask' + "\n"
            mask = np.transpose(masks[this_mask])
            mask_info = mask_plotting_information(mask, excluded_color=excluded_color)
            plt.close('all')
            fig, ax = plt.subplots()
            im = ax.imshow(mask, origin='lower', cmap=cm.Greys)
            ax.set_xlabel('solar X')
            ax.set_ylabel('solar Y')
            ax.set_title("{:s}{:s}{:s}".format(super_title, description, mask_info))
            ax.grid(linestyle=":")
            filename = f'spatial.mask.{this_mask}.{base_filename}.png'
            filepath = os.path.join(directory, filename)
            plt.tight_layout()
            plt.savefig(filepath)

        ###########################
        # Plot the intensity with and without the combined mask
        masks['none'] = np.zeros_like(masks['combined'])
        for this_mask in ('none', 'combined'):
            description = f'total emission (mask={this_mask})' + "\n"
            data = np.ma.array(total_intensity, mask=np.transpose(masks[this_mask]))
            mask_info = mask_plotting_information(data.mask, excluded_color=excluded_color)

            # Spatial distribution
            plt.close('all')
            fig, ax = plt.subplots()
            norm = ImageNormalize(stretch=AsinhStretch(0.01), vmin=data.min(), vmax=data.max())
            im = ax.imshow(data, origin='lower', cmap=intensity_cmap, norm=norm)
            im.cmap.set_bad(excluded_color)
            ax.set_xlabel('solar X')
            ax.set_ylabel('solar Y')
            ax.set_title("{:s}{:s}{:s}".format(super_title, description, mask_info))
            ax.grid(linestyle=":")
            fig.colorbar(im, ax=ax, label="total emission")
            filename = f'spatial.emission_{this_mask}.{base_filename}.png'
            filepath = os.path.join(directory, filename)
            plt.tight_layout()
            plt.savefig(filepath)

            # Histograms
            # Summary statistics
            compressed = np.log10(data.flatten().compressed())
            compressed = compressed[np.isfinite(compressed)]
            ss = SummaryStatistics(compressed, ci=(0.16, 0.84, 0.025, 0.975), bins=bins)

            # Credible interval strings
            ci_a = "{:.1f}$\%$".format(100*ss.ci[0])
            ci_b = "{:.1f}$\%$".format(100*ss.ci[1])
            ci_c = "{:.1f}$\%$".format(100*ss.ci[2])
            ci_d = "{:.1f}$\%$".format(100*ss.ci[3])
            ci_1 = 'C.I. {:s}$\\rightarrow${:s} ({:.2f}$\\rightarrow${:.2f})'.format(ci_a, ci_b, ss.cred[0], ss.cred[1])
            ci_2 = 'C.I. {:s}$\\rightarrow${:s} ({:.2f}$\\rightarrow${:.2f})'.format(ci_c, ci_d, ss.cred[2], ss.cred[3])

            # Histograms
            description = f"histogram of emission (mask={this_mask})" + "\n"
            mask_info = mask_plotting_information(data.mask)
            plt.close('all')
            fig, ax = plt.subplots()
            h = ax.hist(compressed, bins=bins)
            plt.xlabel('$\log_{10}$(emission)')
            plt.ylabel('number')
            plt.title("{:s}{:s}{:s}".format(super_title, description, mask_info))
            plt.grid(linestyle=":")
            ax.axvline(ss.mean, label='mean ({:.2f})'.format(ss.mean), color='r')
            ax.axvline(ss.mode, label='mode ({:.2f})'.format(ss.mode), color='k')
            ax.axvline(ss.median, label='median ({:.2f})'.format(ss.median), color='y')
            ax.axvline(ss.cred[0], color='r', linestyle=':')
            ax.axvline(ss.cred[1], label=ci_1, color='r', linestyle=':')
            ax.axvline(ss.cred[2], color='k', linestyle=':')
            ax.axvline(ss.cred[3], label=ci_2, color='k', linestyle=':')
            ax.legend()
            filename = f'histogram.emission.{this_mask}.{base_filename}.png'
            filepath = os.path.join(directory, filename)
            plt.savefig(filepath)

        ###########################
        # Plot the results of the fitting
        for i, output_name in enumerate(output_names):
            print(f'Plotting {output_name}')
            for this_mask in ('none', 'combined'):
                print(f'         mask={this_mask}')
                # Transpose because the data is the wrong way around
                data = np.transpose(np.ma.array(outputs[:, :, i], mask=masks[this_mask]))
                compressed = data.flatten().compressed()
                compressed = compressed[np.isfinite(compressed)]

                # The variable name is used in the plot instead of the output_name
                # because we use LaTeX in the plots to match with the variables
                # used in the paper.
                variable_name = df['variable_name'][output_name]

                # Create the title of the plot
                description = f"histogram of {variable_name} (mask={this_mask})" + "\n"
                mask_info = mask_plotting_information(data.mask, excluded_color=excluded_color)
                title = f"{super_title}{description}{mask_info}"

                # Create the histogram plot
                plt.close('all')
                fig, ax = plt.subplots()
                ax = plot_histogram(ax, compressed, bins, variable_name, title)

                # Create the filepath the plot will be saved to, and save it
                filename = f'histogram.{output_name}.{this_mask}.{base_filename}.png'
                filepath = os.path.join(directory, filename)
                print(f'Creating and saving {filepath}')
                plt.savefig(filepath)

                # Spatial distribution
                # Create the title of the plot
                description = f"spatial distribution of {variable_name} (mask={this_mask})" + "\n"
                mask_info = mask_plotting_information(data.mask, excluded_color=excluded_color)
                title = f"{super_title}{description}{mask_info}"

                plt.close('all')
                fig, ax = plt.subplots()
                im, ax = plot_spatial_distribution(ax, data, output_name, title)
                fig.colorbar(im, ax=ax, label=variable_name, extend='max')

                # Create the filepath the plot will be saved to, and save it
                filename = f'spatial.{output_name}.{this_mask}.{base_filename}.png'
                filepath = os.path.join(directory, filename)
                plt.savefig(filepath)

        ###########################
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
        filename = f'sample_fits.{base_filename}.png'
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)


# Gang plots by AIA channel/wave
if 'gang_by_wave' in plots:
    # Gang six plots on one page - ganging by channel
    nrows = 2
    row_size = 5
    ncols = 3
    col_size = 7
    figsize = (ncols*col_size, nrows*row_size)

    # The save directory and base filenames for plots that are not specific to a particular
    # wave (i.e., AIA channel)
    b = [study_type, ds.original_datatype]
    across_waves_directory = ds.datalocationtools.save_location_calculator(ds.roots, b)["project_data"]
    across_waves_base_filename = f"{observation_model_name}_{study_type}_{window}.{power_type}"

    # Create the figures and save them
    for this_mask in ('none', 'combined'):  # Go through the masks we are interested in
        print(f'mask={this_mask}')

        # Spatial emission figures
        plt.close('all')
        efig, eax = plt.subplots(nrows, ncols, figsize=figsize)  # Spatial emission figures

        # Hack to get the output file names - all wavelengths are the same anyway.
        b = [study_type, ds.original_datatype, '335']
        directory = ds.datalocationtools.save_location_calculator(ds.roots, b)["project_data"]
        base_filename = f"{observation_model_name}_{study_type}_335_{window}.{power_type}"
        filename = f'{base_filename}.names.step3.txt'
        filepath = os.path.join(directory, filename)
        print(f'Loading {filepath}')
        with open(filepath) as f:
            output_names = [line.rstrip() for line in f]

        for i, output_name in enumerate(output_names):  # Go through the variables
            print(f'Plotting {output_name}')

            # The variable name is used in the plot instead of the output_name
            # because we use LaTeX in the plots to match with the variables
            # used in the paper.
            variable_name = df['variable_name'][output_name]

            hfig, hax = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)  # Histogram figures
            sfig, sax = plt.subplots(nrows, ncols, figsize=figsize)  # Spatial distribution figures
            for iwave, wave in enumerate(waves):  # Load in each wave
                # Position of the plot in the figure
                this_row = iwave // ncols
                this_col = iwave - this_row * ncols

                # Colour maps for spatial distributions
                if "verify_fitting" not in study_type:
                    intensity_cmap = plt.get_cmap(f'sdoaia{wave}')
                else:
                    intensity_cmap = plt.get_cmap('gray')

                # The super title describes the study type and the wavelength
                super_title = "{:s}, {:s}\n".format(study_type.replace("_", " "), wave)

                # General notification that we have a new data-set
                print('\nLoading New Data')

                # branch location
                b = [study_type, ds.original_datatype, wave]

                # Location of the project data
                directory = ds.datalocationtools.save_location_calculator(ds.roots, b)["project_data"]

                # Base filename
                base_filename = f"{observation_model_name}_{study_type}_{wave}_{window}.{power_type}"

                # Load in some information about how to treat and plot the outputs, for example
                # output_name,lower_bound,upper_bound,variable_name
                # "amplitude_0",None,None,"A_{0}"
                # "alpha_0",0,4,"n"
                filename = 'models.outputs_information.{:s}.csv'.format(observation_model_name)
                df = pd.read_csv(filename, index_col=0)
                df = df.replace({"None": None})

                # Load in the fit parameters
                outputs = load_fit_parameters(directory, base_filename)

                # Load in the fit parameter output names
                output_names = load_fit_parameter_output_names(directory, base_filename)

                # Load in the analysis details
                filename = f'{base_filename}.analysis.step3.npz'
                filepath = os.path.join(directory, filename)
                print(f'Loading {filepath}')
                subsection = np.load(filepath)['arr_0']
                normalize_frequencies = np.all(np.load(filepath)['arr_1'])
                divide_by_initial_power = np.all(np.load(filepath)['arr_2'])

                # Load in the mask data
                masks = load_masks(directory, base_filename)
                masks['none'] = np.zeros_like(masks['combined'])

                # Mask the data
                data = np.transpose(np.ma.array(outputs[:, :, i], mask=masks[this_mask]))
                compressed = data.flatten().compressed()
                compressed = compressed[np.isfinite(compressed)]

                # Create the title of the plot
                description = f"histogram of {variable_name} (mask={this_mask})" + "\n"
                mask_info = mask_plotting_information(data.mask, excluded_color=excluded_color)
                title = f"{super_title}{description}{mask_info}"

                # Create the histogram plot
                hax[this_row, this_col] = plot_histogram(hax[this_row, this_col], compressed, bins, variable_name, title)

                # Spatial distribution
                # Create the title of the plot
                description = f"spatial distribution of {variable_name} (mask={this_mask})" + "\n"
                mask_info = mask_plotting_information(data.mask, excluded_color=excluded_color)
                title = f"{super_title}{description}{mask_info}"

                # Create the spatial distribution plot
                im, sax[this_row, this_col] = plot_spatial_distribution(sax[this_row, this_col], data, output_name, title)
                sfig.colorbar(im, ax=sax[this_row, this_col], label=variable_name, extend='max')

                # Load in the original time series data
                if i == 0:
                    if "verify_fitting" not in study_type:
                        filename = f'{study_type}_{wave}.step1.npz'
                        filepath = os.path.join(directory, filename)
                        print(f'Loading {filepath}')
                        emission = (np.load(filepath)['arr_0'])[subsection[0]:subsection[1], subsection[2]:subsection[3], :]
                        intensity_cmap = plt.get_cmap(f'sdoaia{wave}')
                    else:
                        emission = np.ones_like(observed)
                        intensity_cmap = plt.get_cmap('gray')
                    total_intensity = np.transpose(np.sum(emission, axis=2))
                    data = np.ma.array(total_intensity, mask=np.transpose(masks[this_mask]))
                    description = f'total emission (mask={this_mask})' + "\n"
                    mask_info = mask_plotting_information(data.mask, excluded_color=excluded_color)
                    title = f"{super_title}{description}{mask_info}"
                    im, eax[this_row, this_col] = plot_emission(eax[this_row, this_col], data, title, intensity_cmap)
                    efig.colorbar(im, ax=eax[this_row, this_col], label="total emission")

            # Save the histograms
            hfig.tight_layout()
            filename = f'histogram.joint.{output_name}.{this_mask}.{across_waves_base_filename}.png'
            filepath = os.path.join(across_waves_directory, filename)
            print(f'Creating and saving {filepath}')
            hfig.savefig(filepath)

            # Save the spatial distributions
            sfig.tight_layout()
            filename = f'spatial.joint.{output_name}.{this_mask}.{across_waves_base_filename}.png'
            filepath = os.path.join(across_waves_directory, filename)
            print(f'Creating and saving {filepath}')
            sfig.savefig(filepath)

        # Save the emission plots
        efig.tight_layout()
        filename = f'emission.joint.{this_mask}.{across_waves_base_filename}.png'
        filepath = os.path.join(across_waves_directory, filename)
        print(f'Creating and saving {filepath}')
        efig.savefig(filepath)


# Gang plots by AIA channel and overplot results from different simulations
if 'gang_by_simulation_and_wave' in plots:
    nrows = 2
    row_size = 5
    ncols = 3
    col_size = 7
    figsize = (ncols*col_size, nrows*row_size)

    study_type_colors = dict()
    study_type_colors['bv_simulation_low_fn'] = 'red'
    study_type_colors['bv_simulation_intermediate_fn'] = 'green'
    study_type_colors['bv_simulation_high_fn'] = 'blue'

    # Labels that will appear in the legend in the plot
    study_type_labels = dict()
    study_type_labels['bv_simulation_low_fn'] = 'low frequency'
    study_type_labels['bv_simulation_intermediate_fn'] = 'intermediate frequency'
    study_type_labels['bv_simulation_high_fn'] = 'high frequency'

    # The save directory and base filenames for plots that are not specific to a particular
    # wave or study type (i.e., AIA channel)
    across_waves_study_directory = (ds.roots)["project_data"]
    across_waves_study_base_filename = f"{observation_model_name}_{window}.{power_type}"

    # Output names
    output_names = get_output_names_hack(study_type, ds, observation_model_name, window, power_type, wave='335')

    for ion, output_name in enumerate(output_names):  # Go through the variables
        print(f'Generating plots for {output_name}.')
        variable_name = df['variable_name'][output_name]

        plt.close('all')
        hfig, hax = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)  # Histogram figures
        pfig, pax = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)  # Probability distribution figures
        kfig, kax = plt.subplots(nrows, ncols, figsize=figsize)  # Joint mask images
        vfig, vax = plt.subplots(nrows, ncols, figsize=figsize)  # Scaled value images
        for iwave, wave in enumerate(waves):
            print(f'Loading wave {wave}.')
            this_row = iwave // ncols
            this_col = iwave - this_row * ncols

            super_title = f"{wave} simulations\n"

            for_histograms = dict()
            for ist, study_type in enumerate(study_types):
                print(f'Loading study type {study_type}.')

                # branch location
                b = [study_type, ds.original_datatype, wave]

                # Location of the project data
                directory = ds.datalocationtools.save_location_calculator(ds.roots, b)["project_data"]

                # Base filename
                base_filename = f"{observation_model_name}_{study_type}_{wave}_{window}.{power_type}"

                # Load in the mask data
                masks = load_masks(directory, base_filename)

                # Load in the fit parameters
                outputs = load_fit_parameters(directory, base_filename)

                # Load in the fit parameter output names
                output_names = load_fit_parameter_output_names(directory, base_filename)

                # Create an RGB image of the masks
                combined_mask = masks['combined']
                if ist == 0:
                    ny = combined_mask.shape[0]
                    nx = combined_mask.shape[1]
                    image_mask = np.zeros((ny, nx, 3))
                    image_scaled = np.zeros_like(image_mask)
                image_mask[:, :, ist] = 1-np.asarray(np.transpose(combined_mask), dtype=int)

                # Scale the input data to a range 0-1 so we can make a RGB blended image
                # image_scaled[:, :, ist] =

                # Mask the data
                data = np.transpose(np.ma.array(outputs[:, :, ion], mask=combined_mask))

                # Create the data used for the histograms
                compressed = data.flatten().compressed()
                compressed = compressed[np.isfinite(compressed)]
                for_histograms[study_type] = compressed

            # Create the plot of the overlaid masks as a single RGB image
            description = f"locations of included fits from each simulation overlaid as RGB triple"
            title = f"{super_title}{description}"
            kax[this_row, this_col] = plot_overlay_image(kax[this_row, this_col], image_mask, title)

            # Create the plot of the scaled output values as a single RGB image
            #vax[this_row, this_col] = overlay_image_plot(vax[this_row, this_col], image_scaled)

            # Create the plot of the overlaid histograms
            description = f"histograms of {variable_name} for each simulation"
            title = f"{super_title}{description}"
            hax[this_row, this_col] = plot_overlay_histograms(hax[this_row, this_col], for_histograms, bins, study_type_colors, study_type_labels, study_types, variable_name, title)

            # Create the plot of the overlaid probability distributions
            description = f"probability densities for {variable_name} for each simulation"
            title = f"{super_title}{description}"
            pax[this_row, this_col] = plot_overlay_histograms(pax[this_row, this_col], for_histograms, bins, study_type_colors, study_type_labels, study_types, variable_name, title, density=True)

        # Save the histograms
        this_mask = 'combined'
        hfig.tight_layout()
        filename = f'histogram.joint.{output_name}.{this_mask}.{across_waves_study_base_filename}.png'
        filepath = os.path.join(across_waves_study_directory, filename)
        print(f'Creating and saving {filepath}')
        hfig.savefig(filepath)

        # Save the mask RGB
        kfig.tight_layout()
        filename = f'mask_rgb.joint.{this_mask}.{across_waves_study_base_filename}.png'
        filepath = os.path.join(across_waves_study_directory, filename)
        print(f'Creating and saving {filepath}')
        kfig.savefig(filepath)

        # Save the probability distributions
        pfig.tight_layout()
        filename = f'probability.joint.{output_name}.{this_mask}.{across_waves_study_base_filename}.png'
        filepath = os.path.join(across_waves_study_directory, filename)
        print(f'Creating and saving {filepath}')
        pfig.savefig(filepath)

        # Save the scaled value RGB
        #vfig.tight_layout()
        #filename = f'scaled_value.joint.{output_name}.{this_mask}.{across_waves_study_base_filename}.png'
        #filepath = os.path.join(across_waves_study_directory, filename)
        #print(f'Creating and saving {filepath}')
        #vfig.savefig(filepath)
