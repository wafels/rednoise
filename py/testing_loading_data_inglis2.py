"""
Testing RN
"""

import re
from reform_lyra_time import reform_lyra_time
from sunpy.time import *
import numpy as np
import matplotlib.pyplot as plt
import rn_utils
import pymc
import os
import pymcmodels
import pyfits
from rebin_prototype import rebin_prototype
from matplotlib import rc
import idlsave

def analyse_event(event_number=None,wavelength='l3'):

    # Use LaTeX in matplotlib - very nice.
    #rc('text', usetex=True)
    # Set up the simulated data
    n = 600
    dt = 1.0
    alpha = 2.0

    directory = os.path.expanduser('~/physics/rednoise_outputs/')
    format = 'png'

    # Where to dump the CSV files
    csv_directory = os.path.expanduser('~/ts/csv/')

    # Calculate a filename
    filename = ""

    # Factor - if true, normalize power by the data variance
    factor = True

    #for seed in range(0,5):
    #   print seed
    #  # get some test data
    #  test_data = rn_utils.simulated_power_law(n, dt, alpha, seed=seed, poisson=False)
    #  rn_utils.write_ts_as_csv(csv_directory,
    #                           filename + '_seed=' + str(seed),
    #                           dt * np.arange(0, len(test_data)),
    #                           test_data)

    #read the event list SAV file if an event number is given
    if event_number:
        event_list=idlsave.read('/Users/ainglis/physics/event_list/event_list.sav',verbose=False)
        lyra_start_time=event_list.event_list.lyra_start_time[0][event_number]
        lyra_end_time=event_list.event_list.lyra_end_time[0][event_number]
        rhessi_start_time=event_list.event_list.hessi_start_time[0][event_number]
        rhessi_end_time=event_list.event_list.hessi_end_time[0][event_number]
        goes_class=event_list.event_list.goes_class[0][event_number]

        #want to convert the start and end times from string to UTIME, so we can find the start and end indices.
        basetime=parse_time('1979-01-01 00:00:00')
        lyrastarttime=reform_lyra_time(lyra_start_time)
        lst=parse_time(lyrastarttime)
        #get start time in seconds since 1st Jan 1979
        lst1979=(lst-basetime).total_seconds()
        lyraendtime=reform_lyra_time(lyra_end_time)
        let=parse_time(lyraendtime)
        #get end time in seconds since 1st Jan 1979
        let1979=(let-basetime).total_seconds()


        #now we need to load the correct LYRA or RHESSI data file and select the correct time intervals.
        #For rhessi, one file per event so list obs_summ* files and load file[event_number] from the list

        #find the lyra file
        dir='/Users/ainglis/physics/event_list/'
        allfiles=os.listdir('/Users/ainglis/physics/event_list/')
        lyra_files=[file for file in allfiles if file.startswith('lyra')]

        #For LYRA, not as many data files as events! Need to make sure we open the correct file!
        #Do some bad python to find the right file to open. Parse the filenames into dates and compare with the date of the lyra start time
        lyra_file_date_strings=[]
        lyra_file_date_objects=[]
        for file in lyra_files:
            tmpstring=re.split('-',file)
            tmpdatestr=tmpstring[0]
            tmpyr=tmpdatestr[5:9]
            tmpmnth=tmpdatestr[9:11]
            tmpday=tmpdatestr[11:13]
            lyra_file_date_string=tmpyr + '-' + tmpmnth + '-' +tmpday
            lyra_file_date_strings.append(lyra_file_date_string)
            tmptime=parse_time(lyra_file_date_string)
            lyra_file_date_objects.append(tmptime)

        #find the index of the correct file to load
        ind=[]
        for o in lyra_file_date_objects:
            #if the file date string matches the lyra start time date string then use this file
            ind.append(o.date() == lst.date())
        #but what if the event runs across 00:00? Need to load a second file in this case - worry about this later
        
        lyra_select=lyra_files[ind.index(True)]
        lyra_file=pyfits.open(dir+lyra_select)
        lyra_header=lyra_file[0].header
        lyra_data=lyra_file[1].data
        lyra_time=lyra_data.field(0)
        l3=lyra_data.field(3)
        l4=lyra_data.field(4)
        dt=lyra_time[100]-lyra_time[99]

        #find the rhessi file
        rdir='/Users/ainglis/physics/event_list/event_list_data/'
        allrfiles=os.listdir('/Users/ainglis/physics/event_list/event_list_data/')
        rhessi_files=[file for file in allrfiles if file.startswith('obs_summ')]
        rhessi_select=rhessi_files[event_number]
        rhessi_file=idlsave.read(rdir+rhessi_select,verbose=False)
        rhessi_header=rhessi_file.obs_info
        rhessi_data=rhessi_file.obs_data
        rhessi_time=rhessi_file.obs_times

        r612=rhessi_data[:,1]
        r1225=rhessi_data[:,2]
        r2550=rhessi_data[:,3]
        r50100=rhessi_data[:,4]
        

        #find the total number of seconds since 1st Jan 1979 up to the start of the day
        s=lst.date()-basetime.date()
        secs=s.total_seconds()
        #finally(!) get the lyra time array in seconds since 1st Jan 1979
        lyt1979=secs+lyra_time

        #find the start and end time indices for LYRA
        lyt_start_ind=np.searchsorted(lyt1979,lst1979)
        lyt_end_ind=np.searchsorted(lyt1979,let1979)
        
        #get rhessi time objects too
        rhessistarttime=reform_lyra_time(rhessi_start_time)
        rst=parse_time(rhessistarttime)
        #get start time in seconds since 1st Jan 1979
        rst1979=(rst-basetime).total_seconds()
        rhessiendtime=reform_lyra_time(rhessi_end_time)
        ret=parse_time(rhessiendtime)
        #get end time in seconds since 1st Jan 1979
        ret1979=(ret-basetime).total_seconds()

        #find the start and end time indices for RHESSI
        r_start_ind=np.searchsorted(rhessi_time,rst1979)
        r_end_ind=np.searchsorted(rhessi_time,ret1979)
        

        #work out a date string to append files
        strings1=re.split('_',rhessi_select)
        day=strings1[2]
        strings2=re.split('\.',strings1[3])
        tim=strings2[0]
        date_string=day+'_'+tim
    if not event_number:
    
        #import some real LYRA data manually
        seed=0
        lyra_file=pyfits.open('/Users/ainglis/physics/event_list/lyra_20120908-000000_lev2_std.fits')
        lyra_header=lyra_file[0].header
        lyra_data=lyra_file[1].data
        lyra_time=lyra_data.field(0)
        l3=lyra_data.field(3)
        l4=lyra_data.field(4)
        dt=lyra_time[100]-lyra_time[99]

        #import RHESSI data manually too
        rhessi_file=idlsave.read('/Users/ainglis/physics/event_list/event_list_data/obs_summ_20110607_061600.sav')
        rhessi_header=rhessi_file.obs_info
        rhessi_data=rhessi_file.obs_data
        rhessi_time=rhessi_file.obs_times

        r612=rhessi_data[:,1]
        r1225=rhessi_data[:,2]
        r2550=rhessi_data[:,3]
        r50100=rhessi_data[:,4]

    #choose the wavelength to analyse based on keyword input
    #if LYRA chosen, rebin to 1s intervals to improve S/N
    if wavelength == 'l3':
        test_data=rebin_prototype(l3[lyt_start_ind:lyt_end_ind],20)
        dt=dt*20
    if wavelength == 'l4':
        test_data=rebin_prototype(l4[lyt_start_ind:lyt_end_ind],20)
        dt=dt*20

    if wavelength == '612':
        test_data=r612[r_start_ind:r_end_ind]
        dt=4
    if wavelength == '1225':
        test_data=r1225[r_start_ind:r_end_ind]
        dt=4
    if wavelength == '2550':
        test_data=r2550[r_start_ind:r_end_ind]
        dt=4
    if wavelength == '50100':
        test_data=r50100[r_start_ind:r_end_ind]
        dt=4
    
    n=len(test_data)

    # get the power spectrum and frequencies we will analyze at
    if factor:
        data_norm = n * np.var(test_data)
        print(data_norm)
    else:
        data_norm = 1.0
    observed_power_spectrum = ((np.absolute(np.fft.fft(test_data))) ** 2)
    fftfreq = np.fft.fftfreq(n, dt)
    analysis_frequencies = fftfreq[fftfreq >= 0][1:-1]
    analysis_power = observed_power_spectrum[fftfreq >= 0][1:-1]

    # get a simple estimate of the power law index
    estimate = rn_utils.do_simple_fit(analysis_frequencies, analysis_power)
    c_estimate = estimate[0]
    m_estimate = estimate[1]

    # plot the power law spectrum
    power_fit = c_estimate * analysis_frequencies ** (-m_estimate)

    # Define the MCMC model
    #this_model = pymcmodels.single_power_law(analysis_frequencies, analysis_power, m_estimate)
    this_model=pymcmodels.broken_power_law(analysis_frequencies,analysis_power,1.0,m_estimate,0.0,-1.0)
    #this_model=pymcmodels.power_law_with_constant(analysis_frequencies,analysis_power,10,m_estimate,0.001)
    # Set up the MCMC model
    M1 = pymc.MCMC(this_model)

    # Run the sampler
    M1.sample(iter=100000, burn=1000, thin=10, progress_bar=False)

    # Get the power law index before the break and save the results
    #pli = M1.trace("power_law_index")[:]
    pli=M1.trace("delta1")[:]
    s_pli = rn_utils.summary_stats(pli, 0.05, bins=40)
    pi95 = rn_utils.credible_interval(pli, ci=0.95)

    cli = M1.trace("power_law_norm")[:]
    s_cli = rn_utils.summary_stats(cli, 0.05, bins=40)
    ci95 = rn_utils.credible_interval(cli, ci=0.95)

    bayes_mean_fit = np.exp(s_cli["mean"]) * (analysis_frequencies ** -s_pli["mean"])
    bayes_mode_fit = np.exp(s_cli["mode"]) * (analysis_frequencies ** -s_pli["mode"])
    bayes_c68l_fit = np.exp(s_cli["ci68"][0]) * (analysis_frequencies ** -s_pli["ci68"][0])
    bayes_c68u_fit = np.exp(s_cli["ci68"][1]) * (analysis_frequencies ** -s_pli["ci68"][1])
    bayes_c95l_fit = np.exp(ci95[0]) * (analysis_frequencies ** -pi95[0])
    bayes_c95u_fit = np.exp(ci95[1]) * (analysis_frequencies ** -pi95[1])



    #find out where the power law break is
    f=M1.trace("breakf")[:]
    fb=rn_utils.summary_stats(f,0.05,bins=40)
    #f95=rn_utils.credible_interval(fb,ci=95)

    #find the frequency index where the break is located
    floc=np.searchsorted(np.log10(analysis_frequencies),fb["mean"])

    # Get the power law index for after the break and save the results
    #pli = M1.trace("power_law_index")[:]
    plib=M1.trace("delta2")[:]
    s_plib = rn_utils.summary_stats(plib, 0.05, bins=40)
    pi95b = rn_utils.credible_interval(plib, ci=0.95)

    #mod=(analysis_frequencies[floc]-analysis_frequencies[0])*s_plib["mean"]

    bayes_mean_fitb = np.exp(s_cli["mean"]) * (analysis_frequencies[floc] ** (-(s_pli["mean"] - s_plib["mean"]))) * (analysis_frequencies[floc:] ** -s_plib["mean"])
    bayes_mode_fitb = np.exp(s_cli["mode"]) * (analysis_frequencies[floc] ** (-(s_pli["mode"] - s_plib["mode"])))  * (analysis_frequencies[floc:] ** -s_plib["mode"])
    bayes_c68l_fitb = np.exp(s_cli["ci68"][0]) * (analysis_frequencies[floc] ** (-(s_pli["ci68"][0] - s_plib["ci68"][0]))) * (analysis_frequencies[floc:] ** -s_plib["ci68"][0])
    bayes_c68u_fitb = np.exp(s_cli["ci68"][1]) * (analysis_frequencies[floc] ** (-(s_pli["ci68"][1] - s_plib["ci68"][1]))) * (analysis_frequencies[floc:] ** -s_plib["ci68"][1])
    bayes_c95l_fitb = np.exp(ci95[0]) * (analysis_frequencies[floc] ** (-(pi95[0] - pi95b[0]))) * (analysis_frequencies[floc:] ** -pi95b[0])
    bayes_c95u_fitb = np.exp(ci95[1]) * (analysis_frequencies[floc] ** (-(pi95[1] - pi95b[1]))) * (analysis_frequencies[floc:] ** -pi95b[1])


    # plot the power spectrum, the quick fit, and the Bayesian fit
    plt.figure(1)
    plt.plot(test_data)
    plt.xlabel('time (seconds); sample cadence = %4.2f second' % (dt))
    plt.ylabel('emission')
    if wavelength == 'l3':
        plt.title(r'LYRA Al filter')
    if wavelength == 'l4':
        plt.title(r'LYRA Zr filter')
    if wavelength == '612':
        plt.title(r'RHESSI 6-12 keV')
    if wavelength == '1225':
        plt.title(r'RHESSI 12-25 keV')
    if wavelength == '2550':
        plt.title(r'RHESSI 25-50 keV')
    if wavelength == '50100':
        plt.title(r'RHESSI 50-100 keV')
    plt.savefig(directory + filename + 'timeseries_'+ wavelength +'_' + date_string+'.png', format=format)
    plt.close()

    plt.figure(2)
    plt.loglog(analysis_frequencies, analysis_power / data_norm, label=r'observed power')   #: $\alpha_{true}= %4.2f$' % (alpha))
    # plt.loglog(analysis_frequencies, power_fit, label=r'$\alpha_{lstsqr}=%4.2f$' % (m_estimate))
    #plt.loglog(analysis_frequencies, bayes_mean_fit / data_norm, label=r'$\overline{\alpha}=%4.2f$' % (s_pli["mean"]))
    plt.loglog(analysis_frequencies[:floc], bayes_mean_fit[:floc] / data_norm, label=r'$\alpha_{mean}=%4.2f$' % (s_pli["mean"]))
    plt.loglog(analysis_frequencies[floc:], bayes_mean_fitb / data_norm, label=r'$\alpha_{mean}=%4.2f$' % (s_plib["mean"]))
    plt.loglog(analysis_frequencies[floc], bayes_mean_fit[floc] / data_norm,label=r'$f_{break}=%4.2f$' % (fb["mean"]))
    #plt.loglog(analysis_frequencies, bayes_c68l_fit / data_norm, label='Lower 68% CI: ' + r'$\alpha=%4.2f$' % (s_pli["ci68"][0]))
    #plt.loglog(analysis_frequencies, bayes_c68u_fit / data_norm, label='Upper 68% CI: ' + r'$\alpha=%4.2f$' % (s_pli["ci68"][1]))
    #plt.loglog(analysis_frequencies, bayes_c95l_fit / data_norm, label='Lower 95% CI: ' + r'$\alpha=%4.2f$' % (pi95[0]))
    #plt.loglog(analysis_frequencies, bayes_c95u_fit / data_norm, label='Upper 95% CI: ' + r'$\alpha=%4.2f$' % (pi95[1]))
    plt.xlabel('frequency')
    plt.axhline(y=1.0, color='black', label=r'expected Gaussian noise value')
    if factor:
        plt.ylabel('Fourier power / data variance')
    else:
        plt.ylabel('Fourier power')
    if wavelength == 'l3':
        plt.title('Broken power law fit (LYRA Al data)')
    if wavelength == 'l4':
        plt.title('Broken power law fit (LYRA Zr data)')
    if wavelength == '612':
        plt.title('Broken power law fit (RHESSI 6-12 keV data)')
    if wavelength == '1225':
        plt.title('Broken power law fit (RHESSI 12-25 keV data)')
    if wavelength == '2550':
        plt.title('Broken power law fit (RHESSI 25-50 keV data)')
    if wavelength == '50100':
        plt.title('Broken power law fit (RHESSI 50-100 keV data)')
        
    plt.legend(loc=3,prop={'size':12})
    plt.savefig(directory + filename + 'fourier_loglog_'+ wavelength +'_' + date_string+'.png', format=format)
    plt.close()

    plt.figure(3)
    plt.loglog(analysis_frequencies, analysis_power/bayes_mean_fit, label=r'observed power / mean fit')
    plt.loglog(analysis_frequencies, analysis_power/bayes_mode_fit, label=r'observed power / mode fit')
    plt.axhline(y=1.0, color='black')
    plt.xlabel('frequency')
    plt.ylabel('power')
    plt.title('data / fit')
    plt.legend(loc=3)
    plt.savefig(directory + filename + 'data_divided_by_fit_'+ wavelength + '_'+date_string+'.png', format=format)
    plt.close()


