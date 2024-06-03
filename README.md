# Repository for the paper on Hidden Multivariate pattern (HMP)

**Title: Trial-by-trial detection of cognitive events in neural time-series**

**Authors : Gabriel Weindel, Leendert van Maanen, Jelmer P. Borst**

![Figure of the paper in 9 panels summarizing the HMP method, see caption of Figure 1 in the paper](plots/method.png)
This repository contains all the data and code to reproduce the analysis and figures presented in the paper.

- Folder plots/ contains the figure of the paper, a python script to reproduce the method plot (method_plot.py) and some GIMP (.xcf) and inkscape (.svg) files used for post-code figure editing
- Folder results/simulations contains the data and code to reproduce the section "Simulations"
- Folder results/replication contains the data and code to reproduce the section "Example application: HMP and decision-making"

In order to reproduce the analysis and figures the easiest way is to install a specific conda environment as follows:

```bash
conda create -n hmp pymc arviz xarray pandas mne bambi seaborn numpyro watermark
conda activate hmp
conda install pip #if not already installed
pip install hmp
```

In case of problems, for an exact replication of the environment on which those analysis were done install the anaconda environment based on the .yml file
```bash
conda env create -f environment.yml
```

Note: Because of size constraints, the content of:
- /results/replication/estimation_files/*
- /results/simulations/simulated/*

is zipped and stored on https://osf.io/29tgr/. The content needs to be unzipped and placed in the appropriate folders in order to run the code, note though that this is not obligatory but recreating these files through the scripts might take quite some time
