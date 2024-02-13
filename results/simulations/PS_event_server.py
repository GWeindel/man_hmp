
# from multiprocessing import Pool
## Importing the package
import hsmm_mvpy as hmp

## Imports and code specific to the simulation (see tutorial 3 and 4 for real data)
import os
import seaborn as sns
import xarray as xr
import numpy as np
import mne
from hsmm_mvpy import simulations
from scipy.stats import gamma
import matplotlib.pyplot as plt
import time
from itertools import combinations_with_replacement, permutations  
from joblib import Parallel, delayed

from scipy.stats import gamma as sp_dist
from hsmm_mvpy.utils import gamma_scale_to_mean,gamma_mean_to_scale
scale_to_mean, mean_to_scale = gamma_scale_to_mean,gamma_mean_to_scale

n_iterations = 500

def run(iteration):
    source_index = [44, 42, 14, 17]
    rng = np.random.default_rng(seed=iteration)#Setting seeed for reproducibility
    true_dur = rng.uniform(10, 200, 1)[0]
    true_freq = 1000/true_dur/2
    version = '44b3500'
    cpus = 1 # For multiprocessing, usually a good idea to use multiple CPUs as long as you have enough RAM
    path = os.path.join('simulated/')#Where simulated data will go, create that folder if you don't have it where you're executing the code

    #EEG specific
    info = simulations.simulation_info()
    all_other_chans = range(len(info.ch_names[:-61]))#non-eeg
    chan_list = list(np.arange(len(info.ch_names)))
    chan_list = [e for e in chan_list if e not in all_other_chans]
    chan_list.pop(52)#Bad elec
    info = mne.pick_info(info, sel=chan_list)

    #Fixed simulation parameters
    amplitude = 3e-7
    path_to_res = 'event_results/%s' %amplitude
    exists = os.path.exists(path_to_res)
    if not exists:
        os.makedirs(path_to_res)
    results_filename = '/results_%s_%s.nc'%(version,iteration)
    if not os.path.exists(path_to_res+result_filename):
        n_events = 3
        name_sources = simulations.available_sources()[np.random.choice(source_index,n_events+1)]
        n_trials = 100#Number of trials to simulate
        sfreq = 500#5/(1/true_freq/2)#Ensures 5 points per true_freq
        n_comp = 5
        distribution = 'gamma'
        shape = 2
        #Variable simulation parameters

        durations = [10,20,40,50,80,100]

        #Possible combinations
        from itertools import product
        all_combinations = list(durations)

        results = xr.Dataset(data_vars=dict(
                    names = ('event', name_sources[:-1]),
                    test_n_events = ('all_combination', np.zeros(len(all_combinations))*np.nan),
                    loglikelihood = ('all_combination', np.zeros(len(all_combinations))*np.nan),
                    time = ('all_combination', np.zeros(len(all_combinations))*np.nan),
                    hit = (['all_combination', 'event'], np.zeros((len(all_combinations), n_events))*np.nan),
                    false_alarm = ('all_combination', np.zeros(len(all_combinations))*np.nan),
                    true_trial_times = (['all_combination','trial','stage'], np.zeros((len(all_combinations), n_trials, n_events+1))*np.nan),
                    test_trial_times = (['all_combination','trial','stage'], np.zeros((len(all_combinations), n_trials, n_events+1))*np.nan),
                    gen_mags = (['all_combination','event', 'component'], np.zeros((len(all_combinations), n_events, n_comp))*np.nan),
                    recov_mags = (['all_combination','event', 'component'], np.zeros((len(all_combinations), n_events, n_comp))*np.nan),
                ),
                coords=dict(gen_dur = ('all_combination', np.repeat(true_dur,len(durations))),
                            test_dur = ('all_combination', durations)),

                attrs = dict(amplitude=amplitude,
                            n_events=n_events,
                            sfreq=sfreq,
                            n_comp=n_comp,
                            distribution=distribution,
                            shape=shape,
                            n_trials=n_trials,
                            seed=iteration))

        i = 0
        j = 0
        sim_n_trials = 0
        x = 0
        means_list = np.repeat(200,n_events+1) 
        # Function used to generate the data
        times = mean_to_scale(means_list, shape)
        sources = []
        for source in range(len(name_sources)):
            sources.append([name_sources[source], true_freq, amplitude, \
                    sp_dist(shape, scale=times[source])])
        try:            #MNE prints ValueError: stc must have at least three time points, got 2
                 #Perform simulations
    #                 print(times)
            file = simulations.simulate(sources, n_trials, cpus, '%s_%s_%s_%s_%s'%(true_freq,iteration, amplitude, n_trials, version), path=path, 
                                overwrite=False, sfreq=sfreq, seed=iteration)
        except:
            continue
        #Recover info from simulation
        generating_events = np.load(file[1])
        resp_trigger = int(np.max(np.unique(generating_events[:,2])))#Resp trigger is the last source in each trial
        event_id = {'stimulus':1}#trigger 1 = stimulus
        resp_id = {'response':resp_trigger}#Response is defined as the last trigger in a sequence of events
        #Keeping only stimulus and response triggers
        events = generating_events[(generating_events[:,2] == 1) | (generating_events[:,2] == resp_trigger)]#only retain stimulus and response triggers

        number_of_sources = len(np.unique(generating_events[:,2])[1:])#one trigger = one source
        epoch_data = hmp.utils.read_mne_data(file[0], event_id=event_id, resp_id=resp_id, 
                        sfreq=sfreq, events_provided=events, verbose=False, lower_limit_RT=.025*n_events)
        sim_n_trials = len(epoch_data.epochs.values)
        print(sim_n_trials)
        if sim_n_trials == n_trials:
            x += 1
            hmp_data = hmp.utils.transform_data(epoch_data, apply_standard=False, n_comp=n_comp)
            # Computing a true HMP model
            true_init = hmp.models.hmp(hmp_data, sfreq=sfreq, event_width=true_dur, cpus=cpus, distribution=distribution, shape=shape, location=0)#Initialization of the model
            #Recover the actual time of the simulated events
            random_source_times, true_pars, true_amplitudes, true_activities = simulations.simulated_times_and_parameters(generating_events, true_init)
            true_estimates = true_init.fit_single(number_of_sources-1, parameters = true_pars, magnitudes=true_amplitudes, maximization=False)

            for test_event_width in durations:

                print(test_event_width)
                results.true_trial_times.loc[j] = random_source_times
                results.gen_mags.loc[j] = true_amplitudes
                # Estimating an HMP model
                tstart = time.time()
                test_init = hmp.models.hmp(hmp_data, sfreq=sfreq, event_width=test_event_width, cpus=cpus, distribution=distribution, shape=shape)#Initialization of the model

                fit = test_init.fit(verbose=False)
                if fit is not None:
                    tstop = time.time()
                    lkh = fit.likelihoods.values
                    correct_event_capture, corresponding_index_event = simulations.classification_true(fit, true_estimates)
                    n_events_iter = int(np.sum(np.isfinite(fit.magnitudes.values[:,0])))
                    results.test_n_events.loc[j] = n_events_iter
                    results.loglikelihood.loc[j] = lkh
                    results.time.loc[j] = tstop - tstart
                    results.false_alarm.loc[j] = np.max(n_events_iter - len(correct_event_capture),0)
                    test_times = test_init.compute_times(test_init, fit, duration=False, add_rt=True).values
                    if len(correct_event_capture) > 0:
                        index = np.hstack((correct_event_capture,-1))
                        print(correct_event_capture)
                        results.hit.loc[j,correct_event_capture] = correct_event_capture
                        print(corresponding_index_event)
                        print(fit.magnitudes.values)
                        results.test_trial_times.loc[j,:, index] = np.hstack([test_times[:,corresponding_index_event], test_times[:,-1][np.newaxis].T])
                        results.recov_mags.loc[j, correct_event_capture] = fit.magnitudes.values[corresponding_index_event]
                else:
                    results.test_n_events.loc[j] = 0
                print(results.sel(all_combination=j))
                j += 1
            i += 1
            results.to_netcdf(path_to_res+result_filename)

Parallel(40)(delayed(run)(i) for i in np.arange(n_iterations))
