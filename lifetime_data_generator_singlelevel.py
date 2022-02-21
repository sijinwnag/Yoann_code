# %%--  Imports
import sys
# import the function file from another folder:
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\Yoann_code\DPML')
# sys.path.append(r'C:\Users\budac\Documents\GitHub\Yoann_code\DPML')
from Si import *
from main import *
from utils import *
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
import os
# %%-

# %%--  Inputs
SAVEDIR = r"C:\Users\sijin wang\Documents\GitHub\Yoann_code\Savedir_example"
# SAVEDIR = r"C:\Users\budac\Documents\GitHub\Yoann_code\example\Savedir_example" # create a folder that save the output for DPML
TEMPERATURE = [200,250,300,350,400] # define a list of temperature for lifetime data generation (units are in K), the code will not work if your temperature is above 400K
DOPING = [1e15,1e15,1e15,1e15,1e15] # define a list of doping levels for lifetime data generation (units are in cm3)
WAFERTYPE = 'p' # defien the doping type of the wafer for lifetime data generation
NAME = 'Main' # Name of the experiment.
# %%-
# %%--  Hyper-parameters of the experiment
PARAMETERS = {
    'name': NAME,
    'save': False,   # True to save a copy of the printed log, the outputed model and data
    'logML': False,   #   Log the output of the console to a text file
    'n_defects': 8000, # Size of simulated defect data set for machine learning
    'dn_range' : np.logspace(13,17,100),# Number of points to interpolate the curves on, dn is the excess carrier concentration
    'classification_training_keys': ['bandgap_all'], # for  prediction: the name of the columns in dataset that we are going to do classification on
    'regression_training_keys': ['Et_eV_upper','Et_eV_lower','logk_all'], # for prediction: the name of the columns in dataset that we are going to do regression on
    'non-feature_col':['Name', 'Et_eV', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp', 'logk', 'bandgap'] # columns to remove from dataframe in ML training
}
# %%-

# %%-- Generate the data.
exp = Experiment(SaveDir=SAVEDIR, Parameters=PARAMETERS) # define an experiment by defining its save direction and the experiment parameters
exp.updateParameters({'type':WAFERTYPE,'temperature':TEMPERATURE,'doping':DOPING}) # set the wafer type, temperature, and the doping levels of the experiments.
exp.generateDB() # generate the lifetime data for the given experiment
# %%-

# %%--  Export data
exp.exportDataset()
# %%-
