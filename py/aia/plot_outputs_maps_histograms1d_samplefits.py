import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
import matplotlib.cm as cm
import matplotlib.colors as colors
from astropy.visualization import AsinhStretch, ImageNormalize
import sunpy.visualization.colormaps as scm
from tools.statistics import SummaryStatistics
import details_study as ds

parser = argparse.ArgumentParser(description='Plot maps and histograms of results from a channel')
parser.add_argument('-w', '--waves', help='comma separated list of channels', type=str)
parser.add_argument('-s', '--study', help='comma separated list of study types', type=str)
args = parser.parse_args()
waves = [item for item in args.waves.split(',')]

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

# Colour maps
if "verify_fitting" not in study_type:
    intensity_cmap = plt.get_cmap(f'sdoaia{wave}')
else:
    intensity_cmap = plt.get_cmap('gray')

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


def histogram_plot(ax, compressed, bins, variable_name, title):
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

    # Get the summary statistics
    ss = SummaryStatistics(compressed, ci=(0.16, 0.84, 0.025, 0.975), bins=bins)
    # Credible interval strings for the plot
    ci_a = "{:.1f}$\%$".format(100 * ss.ci[0])
    ci_b = "{:.1f}$\%$".format(100 * ss.ci[1])
    ci_c = "{:.1f}$\%$".format(100 * ss.ci[2])
    ci_d = "{:.1f}$\%$".format(100 * ss.ci[3])
    ci_1 = 'C.I. {:s}$\\rightarrow${:s} ({:.2f}$\\rightarrow${:.2f})'.format(ci_a, ci_b, ss.cred[0],
                                                                             ss.cred[1])
    ci_2 = 'C.I. {:s}$\\rightarrow${:s} ({:.2f}$\\rightarrow${:.2f})'.format(ci_c, ci_d, ss.cred[2],
                                                                             ss.cred[3])

    # Create and return the plot
    h = ax.hist(compressed, bins=bins)
    ax.set_xlabel(variable_name)
    ax.set_ylabel('number')
    ax.set_title(title)
    ax.grid(linestyle=":")
    ax.axvline(ss.mean, label='mean ({:.2f})'.format(ss.mean), color='r')
    ax.axvline(ss.mode, label='mode ({:.2f})'.format(ss.mode), color='k')
    ax.axvline(ss.median, label='median ({:.2f})'.format(ss.median), color='y')
    ax.axvline(ss.cred[0], color='r', linestyle=':')
    ax.axvline(ss.cred[1], label=ci_1, color='r', linestyle=':')
    ax.axvline(ss.cred[2], color='k', linestyle=':')
    ax.axvline(ss.cred[3], label=ci_2, color='k', linestyle=':')
    ax.legend()
    return ax


def spatial_distribution_plot(ax, data, output_name, title):
    """
    Creates a spatial distribution plot.

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
    return ax


# Load the data
for wave in waves:
    # General notification that we have a new data-set
    print('\nLoading New Data')

    # branch location
    b = [study_type, ds.original_datatype, wave]

    # Region identifier name
    region_id = ds.datalocationtools.ident_creator(b)

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

    # Load in the original time series data to create an intensity mask
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
            ax = histogram_plot(ax, compressed, bins, variable_name, title)

            """
            # Summary statistics
            ss = SummaryStatistics(compressed, ci=(0.16, 0.84, 0.025, 0.975), bins=bins)
            # Credible interval strings
            ci_a = "{:.1f}$\%$".format(100*ss.ci[0])
            ci_b = "{:.1f}$\%$".format(100*ss.ci[1])
            ci_c = "{:.1f}$\%$".format(100*ss.ci[2])
            ci_d = "{:.1f}$\%$".format(100*ss.ci[3])
            ci_1 = 'C.I. {:s}$\\rightarrow${:s} ({:.2f}$\\rightarrow${:.2f})'.format(ci_a, ci_b, ss.cred[0], ss.cred[1])
            ci_2 = 'C.I. {:s}$\\rightarrow${:s} ({:.2f}$\\rightarrow${:.2f})'.format(ci_c, ci_d, ss.cred[2], ss.cred[3])

            h = ax.hist(compressed, bins=bins)
            plt.xlabel(variable_name)
            plt.ylabel('number')
            plt.title(title)
            plt.grid(linestyle=":")
            ax.axvline(ss.mean, label='mean ({:.2f})'.format(ss.mean), color='r')
            ax.axvline(ss.mode, label='mode ({:.2f})'.format(ss.mode), color='k')
            ax.axvline(ss.median, label='median ({:.2f})'.format(ss.median), color='y')
            ax.axvline(ss.cred[0], color='r', linestyle=':')
            ax.axvline(ss.cred[1], label=ci_1, color='r', linestyle=':')
            ax.axvline(ss.cred[2], color='k', linestyle=':')
            ax.axvline(ss.cred[3], label=ci_2, color='k', linestyle=':')
            ax.legend()
            """

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
            ax = spatial_distribution_plot()
            if output_name == 'alpha_0':
                cmap = cm.Dark2_r
                im = ax.imshow(data, origin='lower', cmap=cmap, norm=colors.Normalize(vmin=0.0, vmax=4.0, clip=False))
                im.cmap.set_over('lemonchiffon')
            elif "err_" in output_name:
                cmap = cm.plasma
                im = ax.imshow(data, origin='lower', cmap=cmap, norm=colors.LogNorm(vmin=compressed.min(), vmax=compressed.max()))
            else:
                cmap = cm.plasma
                im = ax.imshow(data, origin='lower', cmap=cmap)
            im.cmap.set_bad(excluded_color)
            ax.set_xlabel('solar X')
            ax.set_ylabel('solar Y')
            ax.set_title(title)
            ax.grid(linestyle=":")
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


# Gang six plots on one page
hfig, hax = plt.subplots(3, 2)  # Histogram figures
sfig, sax = plt.subplots(3, 2)  # Spatial distribution figures


for this_mask in ('none', 'combined'):  # Go through the masks we are interested in
    print(f'mask={this_mask}')
    for i, output_name in enumerate(output_names):  # Go through the variables
        print(f'Plotting {output_name}')
        for wave in waves:  # Load in each wave
