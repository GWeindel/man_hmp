import sys
sys.path.insert(0, "/home/gweindel/owncloud/projects/RUGUU/main_hmp/hsmm_mvpy/src")

import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mne import channels
from scipy.stats import gamma as sp_gamma
from scipy.stats import invgamma as sp_invgamma
import matplotlib.pyplot as plt
import hsmm_mvpy as hmp
from hsmm_mvpy import simulations

n_trials = 2 #Number of trials to simulate
sfreq = 1000
cpus=10
##### Here we define the sources of the brain activity (event) for each trial
n_events = 3
n_stages = n_events+1
frequency = 10. #Frequency of the event defining its duration, half-sine of 10Hz = 50ms
amplitude = .25e-6 #Amplitude of the event in nAm, defining signal to noise ratio
shape = 2 #shape of the gamma distribution
means = np.array([[50, 100, 200, 100],
                [50, 100, 200, 100]]) #Mean duration of the stages in ms
names = simulations.available_sources()[[0,5,10,15]] #Which source to activate at each stage (see atlas when calling simulations.available_sources())

sources = []
for source in zip(names, means.T): #One source = one frequency, one amplitude and a given by-trial variability distribution
    sources.append([source[0], frequency, amplitude, sp_gamma(shape, scale=source[1])])
# Function used to generate the data
file = simulations.simulate(sources,  n_trials, cpus, 'sim_plots', overwrite=True, sfreq=sfreq, times=means,
                            seed=220)

positions = simulations.simulation_info()
positions = mne.pick_info(positions, sel=[321,318, 359, 366])

events = np.load(file[1])
resp_trigger = int(np.max(np.unique(events[:,2])))#Resp trigger is the last source in each trial
event_id = {'stimulus':1}#trigger 1 = stimulus
resp_id = {'response':resp_trigger}

eeg_data = hmp.utils.read_mne_data(file[0], event_id=event_id, resp_id=resp_id, sfreq=sfreq, 
            events_provided=events, verbose=False, pick_channels=(positions.ch_names))

 
hmp_data = hmp.utils.transform_data(eeg_data, apply_standard=False, method=None)
init = hmp.models.hmp(hmp_data, epoch_data=eeg_data, cpus=1)#Initialization of the model
random_source_times, true_pars, true_magnitudes, true_activities = simulations.simulated_times_and_parameters(events, init)

estimates = init.fit_single(n_events, parameters=true_pars, magnitudes=true_magnitudes, maximization=False)
hmp.visu.plot_topo_timecourse(eeg_data, estimates, positions, init, magnify=1, sensors=True, figsize=(13,1), title='Actual vs estimated bump onsets',
        times_to_display = np.mean(np.cumsum(random_source_times,axis=1),axis=0))

magnitudes, parameters = estimates.magnitudes.values, estimates.parameters.values

gains = np.zeros((init.n_samples, n_events), dtype=np.float64)
for i in range(init.n_dims):
    for trial in range(init.n_trials):
        trial_index = range(init.starts[trial],init.ends[trial]+1)
        gains[trial_index] += init.events[trial_index,i][np.newaxis].T \
            * magnitudes[:,i] - magnitudes[:,i] **2/2
gains = np.exp(gains)
probs = np.zeros([init.max_d,init.n_trials,n_events], dtype=np.float64) # prob per trial
probs_b = np.zeros([init.max_d,init.n_trials,n_events], dtype=np.float64)
for trial in np.arange(init.n_trials):
    # Following assigns gain per trial to variable probs 
    # in direct and reverse order
    probs[:init.durations[trial],trial,:] = \
        gains[init.starts[trial]:init.ends[trial]+1,:] 
    probs_b[:init.durations[trial],trial,:] = \
        gains[init.starts[trial]:init.ends[trial]+1,:][::-1,::-1]

LP = np.zeros([init.max_d, n_events+1], dtype=np.float64) # Gamma pdf for each stage parameters

for stage in range(n_events+1):
    LP[:,stage] = init.distribution_pmf(parameters[stage,0], parameters[stage,1])
BLP = LP[:,::-1] 

forward = np.zeros((init.max_d, init.n_trials, n_events), dtype=np.float64)
backward = np.zeros((init.max_d, init.n_trials, n_events), dtype=np.float64)
# eq1 in Appendix, first definition of likelihood
# For each trial compute gamma pdf * gains
# Start with first bump to loop across others after
forward[:,:,0] = np.tile(LP[:,0][np.newaxis].T,\
    (1,init.n_trials))*probs[:,:,0]
backward[:,:,0] = np.tile(BLP[:,0][np.newaxis].T,\
            (1,init.n_trials)) # reversed Gamma pdf

for bump in np.arange(1,n_events):#continue with other bumps
    add_b = backward[:,:,bump-1]*probs_b[:,:,bump-1]
    for trial in np.arange(init.n_trials):
        temp = np.convolve(forward[:,trial,bump-1], LP[:,bump])
        # convolution between gamma * gains at previous bump and bump
        forward[:,trial,bump] = temp[:init.max_d]
        temp = np.convolve(add_b[:,trial], BLP[:, bump])
        # same but backwards
        backward[:,trial,bump] = temp[:init.max_d]
    forward[:,:,bump] = forward[:,:,bump]*probs[:,:,bump]
for trial in np.arange(init.n_trials):#Undoes inversion
    backward[:init.durations[trial],trial,:] = \
        backward[:init.durations[trial],trial,:][::-1,::-1]
eventprobs = forward * backward


likelihood = np.sum(np.log(eventprobs[:,:,0].sum(axis=0)))#sum over max_samples to avoid 0s in log
eventprobs = eventprobs / np.tile(eventprobs.sum(axis=0), [init.max_d, 1, 1])

import xarray as xr
estimated_times = xr.dot(estimates.eventprobs[:init.n_trials], estimates.eventprobs[:init.n_trials].samples, 
       dims='samples').values[:,:n_events]

ls_bump = {0:'-', 1:'--'}
color_PC = {0:'indianred', 1:'royalblue', 2:'darkorange', 3:'purple'}
color_bump = {0:'gold', 1:'darkgreen', 2:'cornflowerblue',3:'grey'}
ls_bump = {0:'-', 1:'--'}
fig, ax = plt.subplot_mosaic([['a)','b)','b)'],['c)', 'd)', 'e)'], ['f)', 'g)','h)']],
                              layout='constrained',figsize=(12,7), sharex=False,sharey=False, dpi=100)

epoch = 0
for pc in range(init.n_dims):
    sig = hmp_data.sel(epochs=epoch, component=pc).values
    ax['a)'].plot(sig, color=color_PC[pc], label='%s'%(pc+1))
    ax['f)'].plot(init.data_matrix[:,epoch,pc], color=color_PC[pc])
ax['a)'].margins(0, 0.1)
ax['a)'].margins(0, 0.1)
ax['a)'].legend()
# ax['c)'].vlines(np.cumsum(random_source_times[epoch]),-2,2, colors=color_bump.values(), ls='--')

for i in range(1,2):
    j = 0
    for bump in magnitudes:
        ax['g)'].plot(probs[:,0,j], color=color_bump[j], label=f'MVP {j+1}')
        j += 1
ax['g)'].vlines(np.cumsum(random_source_times[epoch]),0,2, colors=color_bump.values(), ls='--')
ax['g)'].legend(frameon=False)

T = 500
stages = true_pars[:-1,1]
for i,stage in enumerate(stages):
    ax['e)'].plot(np.linspace(0,T,1001),sp_gamma.pdf(np.linspace(0,T,1001), 2, scale=stage), 
               label=r'Onset %i, $\theta =$%i'%(i+1,stage), color=color_bump[i])
ax['e)'].set_ylabel(r'$p(t|\theta_i)$')
ax['e)'].legend(frameon=False)

# ax[1].legend()

for i in range(1,2):
    j = 0
    for bump in magnitudes:
        ax['h)'].plot(eventprobs[:,0,j], color=color_bump[j], label=f'Event {j+1}')
        # ax[i,1].vlines(np.sum(random_source_times[i,:j+1]),0,0.03, color=color_bump[j], ls='-', alpha=.7)
        # ax[i,1].vlines(estimated_times[i,j],0,0.03, color=color_bump[j], ls='--', alpha=.7)
        j += 1
ax['h)'].vlines(np.cumsum(random_source_times[epoch]),0,.02, colors=color_bump.values(), ls='--')
ax['h)'].legend(loc='upper center',frameon=False)
# plt.legend()

title_dict = {'a)':'EEG channel timecourses', 'b)':'', 'd)':r'Multivariate matrix $\omega$',
              'e)':r'Event onset distributions Gamma$(2,\theta)$', 'f)':r'Crosscorrelation Pattern * Channel',
              'g)':'Multivariate-pattern timecourses', 'h)':'Event probabilities' , 'c)':'Target Pattern'}
for label, ax_i in ax.items():
    # if label not in [ 'd)']:
    ax_i.set_title(title_dict[label], fontsize=10)
    ax_i.set_title(label, loc='left', fontsize='medium')


ax['a)'].set_ylabel('Channel value')
ax['c)'].set_ylabel(r'Normalized value $\mathbf{H_i}$')
ax['f)'].set_ylabel('Cross-correlated value')
ax['d)'].set_ylabel(r'Channel $c$')
ax['d)'].set_xlabel(r'Event $i$')
ax['g)'].set_ylabel(r'Linear combination $H_t$')
ax['h)'].set_ylabel(r'Convoluted $p(i| \omega_i, \theta_i)$ ')
ax['d)'].set_xticks([.5,1.5,2.5,3.5],[1,2,3,4])
ax['d)'].set_yticks([.5,1.5,2.5,3.5],[1,2,3,4])
        # plt.xlim(0,600)
plt.tight_layout()
magplot = ax['d)'].pcolormesh(true_magnitudes.T,cmap='Spectral_r',vmin=-1, vmax=1)
fig.colorbar(magplot, ax=ax['d)'])
hmp.visu.plot_topo_timecourse(eeg_data, estimates, positions, init, magnify=2, sensors=True, figsize=(13,1),
        times_to_display = np.mean(np.cumsum(random_source_times,axis=1),axis=0), ax=ax['b)'],
                             colorbar=False, event_lines=None, contours=False, 
                             linecolors=color_bump.values())
ax['b)'].set_title('Generative HMP model')
ax['b)'].set_xlabel('Time from stimulus (ms)')
ax['a)'].set_xlabel('Time from stimulus (ms)')
ax['d)'].set_xlabel('MVP')
ax['c)'].set_xlabel('Time (ms)')
ax['e)'].set_xlabel(r'Time $t$ (ms)')
ax['c)'].set_xlabel('Time from stimulus (ms)')
ax['f)'].set_xlabel('Time from stimulus (ms)')
ax['g)'].set_xlabel('Time from stimulus (ms)')
ax['h)'].set_xlabel('Time from stimulus (ms)')


ax['b)'].text(440, 0.15, 'Reaction Time', fontsize=12, rotation='vertical', color=list(color_bump.values())[-1])
ax['c)'].plot(init.template)
plt.savefig('method.pdf',dpi=300,transparent=True,bbox_inches='tight',backend='cairo')
# plt.savefig('method.png',dpi=300,transparent=True,bbox_inches='tight')
plt.show()