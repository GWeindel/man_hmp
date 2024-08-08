
# from multiprocessing import Pool
## Importing the package
import hmp

## Imports and code specific to the simulation (see tutorial 3 and 4 for real data)
import os
import seaborn as sns
import xarray as xr
import numpy as np
import mne
from hmp import simulations
from scipy.stats import gamma
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from scipy.stats import loguniform
from scipy.stats import gamma as sp_dist
from hmp.utils import gamma_scale_to_mean,gamma_mean_to_scale
scale_to_mean, mean_to_scale = gamma_scale_to_mean,gamma_mean_to_scale


path_to_res = 'results_simulation_snr_section3-3/'
exists = os.path.exists(path_to_res)
if not exists:
	os.makedirs(path_to_res)

n_iterations = 500
def run(iteration):
    source_index = [44, 42, 14,17, 59, 49,  25, 23, 34, 47, 21, 11]
    rng = np.random.default_rng(seed=iteration)#Setting seeed for reproducibility
    amplitude = loguniform.rvs(1e-08, 1e-06, size=1, random_state=iteration)[0]#rng.uniform(1e-9, 1000e-9, 1)[0]
    version_sim = '68a8d56'
    version = '2d7a851'
    event_width = 50
    n_trials = 100#Number of trials to simulate
    sfreq = 100
    n_comp = 5
    distribution = 'gamma'
    shape = 2

    #Variable simulation parameters
    #tolerances = [1e-2, 1e-3,1e-4,1e-5]
    n_events = [1,3,5,10]
    
    result_filename = 'results_%s_%s_%s.nc'%(n_trials,version,iteration)
    cpus = 1 # For multiprocessing, usually a good idea to use multiple CPUs as long as you have enough RAM
    path = os.path.join('simulated/1/')#Where simulated data will go, create that folder if you don't have it where you're executing the code

    #EEG specific
    info = simulations.simulation_info()
    all_other_chans = range(len(info.ch_names[:-61]))#non-eeg
    chan_list = list(np.arange(len(info.ch_names)))
    chan_list = [e for e in chan_list if e not in all_other_chans]
    chan_list.pop(52)#Bad elec
    info = mne.pick_info(info, sel=chan_list)

    #Fixed simulation parameters
    frequency = 10.
    path_to_res = 'results_simulation_snr_section3-3/'#%version
    exists = os.path.exists(path_to_res)
    if not exists:
        os.makedirs(path_to_res)
    if not os.path.exists(path_to_res+result_filename):
        #Possible combinations
        from itertools import product
        all_combinations = n_events#list(product(n_events,tolerances))



        results = xr.Dataset(data_vars=dict(
                    means = (['all_combination','stage'], np.zeros((len(all_combinations), max(n_events)+1))*np.nan),
                    amplitude = ('all_combination', np.repeat(amplitude, len(all_combinations))),
                    snr = ('all_combination', np.repeat(np.nan, len(all_combinations))),
                    true_n_events = ('all_combination', np.zeros(len(all_combinations))*np.nan),
                    test_n_events = ('all_combination', np.zeros(len(all_combinations))*np.nan),
                    loglikelihood = ('all_combination', np.zeros(len(all_combinations))*np.nan),
                    time = ('all_combination', np.zeros(len(all_combinations))*np.nan),
                    hit = (['all_combination', 'event'], np.zeros((len(all_combinations), max(n_events)))*np.nan),
                    false_alarm = ('all_combination', np.zeros(len(all_combinations))*np.nan),
                    true_trial_times = (['all_combination','trial','stage'], np.zeros((len(all_combinations), n_trials, max(n_events)+1))*np.nan),
                    test_trial_times = (['all_combination','trial','stage'], np.zeros((len(all_combinations), n_trials, max(n_events)+1))*np.nan),
                    gen_mags = (['all_combination','event', 'component'], np.zeros((len(all_combinations), max(n_events), n_comp))*np.nan),
                    recov_mags = (['all_combination','event', 'component'], np.zeros((len(all_combinations), max(n_events), n_comp))*np.nan),),

                coords=dict(n_events = ('all_combination', np.array(all_combinations))),
                            #tolerances = ('all_combination',  np.array(all_combinations)[:,1])),

                attrs = dict(distribution=distribution,
                             shape=shape,
                             sfreq=sfreq,
                             n_comp=n_comp,
                             event_width=event_width,
                             n_trials=n_trials,
                             seed=iteration))

        i = 0
        all_names = rng.choice(source_index, max(n_events)+1, replace=False)
        for n_ev in n_events:
            name_sources = simulations.available_sources()[all_names[:n_ev+1]]
            means_list = np.repeat(200,n_ev+1) 

            sim_n_trials = 0
            # while sim_n_trials != n_trials:#Sometimes a few trials gets rejected, annoying for the comparison with simulated trials
                # Function used to generate the data
            times = mean_to_scale(means_list, shape)
            sources = []
            for source in range(len(name_sources)):
                sources.append([name_sources[source], frequency, amplitude, \
                          sp_dist(shape, scale=times[source])])
            try: #Some weird untractable error can happen some rare times
                #MNE prints ValueError: stc must have at least three time points, got 2
                    #Perform simulations
        #                 print(times)
                file = simulations.simulate(sources, n_trials, cpus, '%s_%s_%s_%s_%s'%(n_ev, iteration, amplitude, n_trials, version_sim),
                                    path=path, overwrite=False, sfreq=sfreq, seed=iteration, save_snr=True)
            except:
                continue
            snr = np.load(file[2]).mean()
            #Recover info from simulation
            generating_events = np.load(file[1])
            resp_trigger = int(np.max(np.unique(generating_events[:,2])))#Resp trigger is the last source in each trial
            event_id = {'stimulus':1}#trigger 1 = stimulus
            resp_id = {'response':resp_trigger}#Response is defined as the last trigger in a sequence of events
            #Keeping only stimulus and response triggers
            events = generating_events[(generating_events[:,2] == 1) | (generating_events[:,2] == resp_trigger)]#only retain stimulus and response triggers

            number_of_sources = len(np.unique(generating_events[:,2])[1:])#one trigger = one source
            epoch_data = hmp.utils.read_mne_data(file[0], event_id=event_id, resp_id=resp_id, 
                        sfreq=sfreq, events_provided=events, verbose=False)
            sim_n_trials = len(epoch_data.epochs.values)
            if sim_n_trials != n_trials:
                break
                #except: 
                     #continue
            print(sim_n_trials)
            
            hmp_data = hmp.utils.transform_data(epoch_data, apply_standard=False, n_comp=n_comp)
            
            # Computing a true HMP model
            true_init = hmp.models.hmp(hmp_data,epoch_data, sfreq=sfreq, event_width=event_width, cpus=cpus, distribution=distribution, shape=shape, location=0)#Initialization of the model
            #Recover the actual time of the simulated events
            random_source_times, true_pars, true_amplitudes, true_activities = simulations.simulated_times_and_parameters(generating_events, true_init, data=hmp_data.data.T)
            true_estimates = true_init.fit_single(number_of_sources-1, parameters = true_pars, magnitudes=true_amplitudes, maximization=False)

        #       hmp.utils.save_fit(true_estimates, 'event_probs/true_estimates_%s_%s_%s.nc'%(true_combination[0], true_combination[1],iteration))
            results.true_trial_times.loc[i,:,:n_ev+1] = random_source_times
            results.gen_mags.loc[i, :n_ev] = true_amplitudes
            results.true_n_events.loc[i] = n_ev
            results.snr.loc[i] = snr
            # results.names.loc[i, :n_ev+1] = name_sources
            # results.means.loc[i, :n_ev+1] = means_list
            print((n_ev, times, means_list, iteration))

            # Estimating an HMP model
            tstart = time.time()
            test_init = hmp.models.hmp(hmp_data, epoch_data, sfreq=sfreq, event_width=event_width, cpus=cpus, distribution=distribution, shape=shape)#Initialization of the model

            fit = test_init.fit(verbose=False, pval=.05)


            if fit is not None:
                tstop = time.time()
                lkh = fit.likelihoods.values
                #correct_event_capture, corresponding_index_event = simulations.classification_true(fit, true_estimates)
                corresponding_index_event, correct_event_capture = simulations.classification_true(true_init.compute_topologies(epoch_data, true_estimates, true_init, mean=True), test_init.compute_topologies(epoch_data, fit, test_init, mean=True))
                n_events_iter = int(np.sum(np.isfinite(fit.magnitudes.values[:,0])))
                results.test_n_events.loc[i] = n_events_iter
                results.loglikelihood.loc[i] = lkh
                results.time.loc[i] = tstop - tstart
                results.false_alarm.loc[i] = np.max(n_events_iter - len(correct_event_capture),0)
                test_times = test_init.compute_times(test_init, fit, duration=False, add_rt=True).values
                if len(correct_event_capture) > 0:
                    index = np.hstack((correct_event_capture,-1))
                    print(correct_event_capture)
                    results.hit.loc[i,correct_event_capture] = correct_event_capture.astype(int)
                    print(corresponding_index_event)
                    print(fit.magnitudes.values)
                    results.test_trial_times.loc[i,:, index] = np.hstack([test_times[:,corresponding_index_event.astype(int)], test_times[:,-1][np.newaxis].T])

                results.recov_mags.loc[i, correct_event_capture.astype(int)] = fit.magnitudes.values[corresponding_index_event.astype(int)] 
            else:
                results.test_n_events.loc[i] = 0
            i += 1
        results.to_netcdf(path_to_res+result_filename)
Parallel(50)(delayed(run)(i) for i in np.arange(n_iterations))
