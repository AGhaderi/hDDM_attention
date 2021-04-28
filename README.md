# hDDM_attention
Current repository was able to assess the relationship between EEG components and HDDM parameters of top-down attention using multiple regression model


## Citation  

## Description
Consider following stepts to use the repository.
1. use **bahavioral_data** folder in order to filter out rt data  based on the IQR Interquartile range, concatenate subject to a csv file and plot related figures.
2. use **models** folder in order to apply hddm method to compare the five models and plot related figures based on our hypothesis.
3. use **https://github.com/AGhaderi/MNE-Preprocessing** repository to preprocess eeg data
4.  use **eeg** folder to extract contralateral, ipsilateral and neutral amplitude and power, plot related figures and make multiple linear regressions
'
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


