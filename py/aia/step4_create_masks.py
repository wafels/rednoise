"""
Create masks that exclude various parts of the data.
"""
import os
import numpy as np
import pandas as pd
import details_study as ds
from masking import VariableBounds, Fitness, IntensityMask
from tools.statistics import noise_level_estimate


# Which model to look at
observation_model_name = 'pl_c'
window = 'hanning'
power_type = 'absolute'
waves = ['131']


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
    df = pd.read_csv(filename, index_col=0)
    df = df.replace({"None": None})

    # Load in the fit parameters and the output names
    filename = '{:s}_{:s}_{:s}_{:s}.{:s}.outputs.step3.npz'.format(observation_model_name,
                                                                   ds.study_type, wave, window,
                                                                   power_type)
    filepath = os.path.join(directory, filename)
    print(f'Loading {filepath}')
    outputs = np.load(filepath)['arr_0']

    # Load in the names of the output
    filename = '{:s}_{:s}_{:s}_{:s}.{:s}.names.step3.txt'.format(observation_model_name,
                                                                   ds.study_type, wave, window,
                                                                   power_type)
    filepath = os.path.join(directory, filename)
    print(f'Loading {filepath}')
    with open(filepath) as f:
        output_names = [line.rstrip() for line in f]

    # Load in the spectral fits
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

    # Load in the observed Fourier power data
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

    # Calculate different masks. Each mask eliminates results that we do not wish to consider,
    # for example, non-finite values.  Following numpy, True indicates that the value at
    # that position should be masked out.
    # Define the properties of the mask
    mask = np.zeros_like(outputs[:, :, 0], dtype=bool)
    shape = mask.shape
    ny = shape[0]
    nx = shape[1]

    # Calculate the finite mask and the boundaries mask for all the fit outputs
    finite_mask = np.zeros_like(mask)
    bounds_mask = np.zeros_like(mask)
    for i, output_name in enumerate(output_names):
        data = outputs[:, :, i]

        # Finiteness
        is_not_finite = ~np.isfinite(data)
        finite_mask = np.logical_or(finite_mask, is_not_finite)

        # Boundaries
        bounds = (df['lower_bound'][output_name], df['upper_bound'][output_name])
        boundaries = VariableBounds(data, bounds)

        # Update mask
        bounds_mask = np.logical_or(bounds_mask, boundaries.exceeds_low)
        bounds_mask = np.logical_or(bounds_mask, boundaries.exceeds_high)

    # Calculate a fitness mask
    fitness_mask = np.zeros_like(mask)
    for i in range(0, nx):
        for j in range(0, ny):
            fitness = Fitness(observed[i, j, :], mfits[i, j, :], 3)
            fitness_mask[i, j] = not fitness.is_good()

    # Calculate a brightness mask
    total_intensity = np.sum(emission, axis=2)
    noise_level = noise_level_estimate(total_intensity, ((0, 0), (10, 10)))
    intensity_mask = IntensityMask(total_intensity, absolute_level=2*noise_level).mask

    # Collect all the previous masks and combine them
    masks = {"finiteness": finite_mask,
             "bounds": bounds_mask,
             "fitness": fitness_mask,
             "intensity": intensity_mask}
    combined_mask = np.zeros_like(mask)
    for key in list(masks.keys()):
        combined_mask = np.logical_or(combined_mask, masks[key])
    masks['combined'] = combined_mask

    # Save the masks
    for key in list(masks.keys()):
        filename = '{:s}_{:s}_{:s}_{:s}.{:s}.{:s}.step4.npz'.format(observation_model_name,
                                                                    ds.study_type, wave, window,
                                                                    power_type, key)
        filepath = os.path.join(directory, filename)
        print(f'Saving {filepath}')
        np.savez(filepath, masks[key])
