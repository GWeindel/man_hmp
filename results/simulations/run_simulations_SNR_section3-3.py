import os
import numpy as np
from scipy.stats import gamma as sp_dist
from hmp.utils import gamma_mean_to_scale
from joblib import Parallel, delayed
from hmp import simulations



seed = 1234

source_index = [44, 42, 14,17, 59, 49,  25, 23, 34, 47, 21, 11]
amplitudes = [.1e-7, .21e-7, .3e-7]
event_width = 50
n_trials = 1000#Number of trials to simulate
sfreq = 1000
n_comp = 5
distribution = 'gamma'
shape = 2
frequency = 10

#Variable simulation parameters
n_events = [3,5]

cpus = 1 # For multiprocessing, usually a good idea to use multiple CPUs as long as you have enough RAM
path = os.path.join('simulated/')#Where simulated data will go, create that folder if you don't have it where you're executing the code


def run_simulation(n_ev, seed, amplitude, n_trials, frequency, shape, path, sfreq):
    rng = np.random.default_rng(seed=seed)  # Setting seed for reproducibility
    name_sources = rng.choice(simulations.available_sources(), n_ev + 1, replace=False)
    means_list = rng.uniform(50, 300, n_ev + 1)
    times = gamma_mean_to_scale(means_list, shape)
    
    sources = []
    for source in range(len(name_sources)):
        sources.append([
            name_sources[source], frequency, amplitude,
            sp_dist(shape, scale=times[source])
        ])
    
    file = simulations.simulate(sources, n_trials, cpus, 
                                 '%s_%s' % (n_ev, amplitude),
                                 path=path, overwrite=False, sfreq=sfreq, seed=seed, save_snr=True)

results = Parallel(n_jobs=-1)(
    delayed(run_simulation)(n_ev,  seed, amplitude, n_trials, frequency, shape, path, sfreq)
    for n_ev in n_events for amplitude in amplitudes
)