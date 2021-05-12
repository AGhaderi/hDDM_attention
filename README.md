# hDDM_attention
The current repository is for a project named "How spatial attention affects the decision process: looking through the lens of Bayesian hierarchical diffusion model & EEG analysis" at the Shahid Beheshti University, by Mr. Amin-Ghaderi-Kangavari, Dr. Jamal Amani Rad, Prof. Reza Ebrahimpour, and Prof. Kourosh Parand.

## Citation  

## Description
In this project, based on one of the well-known sequential sampling models (SSMs), named the diffusion decision model, we explore the underlying latent process of spatial prioritization in perceptual decision processes, so that for estimating the model parameters (i.e. the drift rate, the boundary separation, and the non-decision time), a Bayesian hierarchical approach is considered, which allows inferences to be done simultaneously in the group and individual level. Moreover, well-established neural components of spatial attention which contributed to the latent process and behavioral performance in a visual face-car perceptual decision are detected based on the event-related potential (ERP) analysis. 

Consider following stepts to use the repository.
1. Use the **bahavioral_data folder** in order to filter out the reaction time (RT) data based on the IQR Interquartile range, concatenate subject to a CSV file, and plot-related figures.
2. Use the **models folder** in order to apply the hierarchical Bayesian drift-diffusion (HDDM) method and compare the cognitive models and plot-related figures based on our hypothesis.
3. Use **https://github.com/AGhaderi/MNE-Preprocessing** repository to preprocess EEG data.
4. Use the **EEG folder** to extract contralateral, ipsilateral, and neutral amplitude and power, plot-related figures, and make multiple linear regressions.

## Data
We re-examine the data from an experiment conducted by ([**here**](https://www.biorxiv.org/content/10.1101/253047v1)). All data used in this research is publicly available in the open science framework [**https://osf.io/q4t8k/**](https://osf.io/q4t8k/).


## Prerequisites

- [hddm](http://ski.clps.brown.edu/hddm_docs/)
- [mne](https://mne.tools/stable/install/mne_python.html)
- [numpy](https://numpy.org/install/)
- [pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
- [matplotlib](https://matplotlib.org/stable/users/installing.html)
- [scipy](https://www.scipy.org/install.html)
- [seaborn](https://seaborn.pydata.org/installing.html)
- [statsmodels](https://www.statsmodels.org/stable/install.html)
- [scikit-learn](https://scikit-learn.org/stable/install.html)


