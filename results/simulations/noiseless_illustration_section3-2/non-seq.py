import simulations
## Importing these packages is specific for this simulation case
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma

## Importing HMP
import hmp

cpus = 1 # For multiprocessing, usually a good idea to use multiple CPUs as long as you have enough RAM

n_trials = 1000 #Number of trials to simulate
sfreq = 1000
##### Here we define the sources of the brain activity (event) for each trial
n_events = 4
frequency = 10. #Frequency of the event defining its duration, half-sine of 10Hz = 50ms
amplitude = 1 #Amplitude of the event in nAm, defining signal to noise ratio
shape = 2 #shape of the gamma distribution

names = ['superiorparietal-lh','inferiortemporal-lh','postcentral-lh','postcentral-rh','superiorparietal-lh']#Which source to activate for each event (see atlas when calling simulations.available_sources())
name_file = 'dataset_non-seq'

means = np.array([50, 100, 250, 600,  350])/shape #Mean duration of the between event times in ms
relations = [1,2,3,1,4] # to which previous event each event is related (1 is stimulus onset)
proportions = [1,1,1,1,1] # How frequent each event is
event_length_samples = [50, 50, 50, 50, 50] # The sampling period of the event (e.g. 100 is full sine)


sources = []
for source in zip(names, means): #One source = one frequency, one amplitude and a given by-trial variability distribution
    sources.append([source[0], frequency,  amplitude, gamma(shape, scale=source[1])])

# Function used to generate the data
file = simulations.simulate(sources, n_trials, cpus, name_file, path='data/', \
            overwrite=False, sfreq=sfreq, noise=False, seed=1234,
            relations=relations, proportions=proportions, 
            event_length_samples=event_length_samples)

#load electrode position, specific to the simulations
positions = simulations.simulation_positions()
