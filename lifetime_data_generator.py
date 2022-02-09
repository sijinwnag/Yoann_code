# %%-- Imports
import sys
# sys.path.append(r'C:\Users\budac\Documents\GitHub\Yoann_code\DPML')
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\Yoann_code\DPML')
from Si import *
# sys.path.append(r'C:\Users\budac\Documents\GitHub\Yoann_code')
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\Yoann_code')
from DPML import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %%-

# %%--  Inputs
SAVEDIR = "savedir_example\\" # you can change this to your own path
FILEPATH = "advanced_example\\data\\sample_original_L.csv"
TEMPERATURE = [158,182,206,230,254,278,300] # below 400K
DOPING = [1.01e16]*len(TEMPERATURE) # make sure T and doping have same length
WAFERTYPE = 'p'
NAME = 'advanced example - multi_level_L'
# %%-

# %%--  Hyper-parameters
PARAMETERS = {
    'name': NAME,
    'save': True,   # True to save a copy of the printed log, the outputed model and data
    'logML':True,   #   Log the output of the console to a text file
    'n_defects': 100, # Size of simulated defect data set for machine learning
    'dn_range' : np.logspace(13,17,100),# Number of points to interpolate the curves on
    'non-feature_col':['Mode','Label',"Name","Et_eV_1","Sn_cm2_1","Sp_cm2_1",'k_1','logSn_1','logSp_1','logk_1','bandgap_1',"Et_eV_2","Sn_cm2_2","Sp_cm2_2",'k_2','logSn_2','logSp_2','logk_2','bandgap_2']
}
PARAM={
        'type': 'p',                #   Wafer doping type
        'Et_min_1':-0.55,             #   Minimum defect energy level
        'Et_max_1':0.55,              #   Maximum defect energy level
        'Et_min_2':-0.55,             #   Minimum defect energy level
        'Et_max_2':0.55,              #   Maximum defect energy level
        'S_min_1':1E-17,              #   Minimum capture cross section
        'S_max_1':1E-13,              #   Maximum capture cross section
        'S_min_2':1E-17,              #   Minimum capture cross section
        'S_max_2':1E-13,              #   Maximum capture cross section
        'Nt':1E12,                  #   Defect density
        'check_auger':True,     #   Check wether to resample if lifetime is auger-limited
        'noise':'',             #   Enable noiseparam
        'noiseparam':0,         #   Adds noise proportional to the log of Delta n
}
# %%-

# %%--  Define experiment: every time remember to run this line to refresh the code.
exp = Experiment(SaveDir=SAVEDIR, Parameters=PARAMETERS)
# %%-

# %%--  Simulate datasets: for a mixture of Two one-level and Single two-level
db_multi=DPML.generateDB_multi(PARAMETERS['n_defects'], TEMPERATURE, DOPING, PARAMETERS['dn_range'], PARAM) # two one-level defect data
db_sah=DPML.generateDB_sah(PARAMETERS['n_defects'], TEMPERATURE, DOPING, PARAMETERS['dn_range'], PARAM) # one two-level defect data
db_multi['Mode']=['Two one-level']*len(db_multi)
db_sah['Mode']=['Single two-level']*len(db_sah)
dataDf=pd.concat([db_multi,db_sah])
dataDf['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf['Mode']]
exp.uploadDB(dataDf)
vocab={
    '0':'Two one-level',
    '1':'Single two-level',
}
# %%-

# %%--  Simulate datasets: for Single two-level
db_multi=DPML.generateDB_multi(PARAMETERS['n_defects'], TEMPERATURE, DOPING, PARAMETERS['dn_range'], PARAM) # two one-level defect data
db_multi['Mode']=['Two one-level']*len(db_multi)
dataDf = db_multi
dataDf['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf['Mode']]
exp.uploadDB(dataDf)
vocab={
    '0':'Two one-level',
    '1':'Single two-level',
}
# %%-

# %%-- Simualate datasets: for Two one-level
db_sah=DPML.generateDB_sah(PARAMETERS['n_defects'], TEMPERATURE, DOPING, PARAMETERS['dn_range'], PARAM) # one two-level defect data
db_sah['Mode']=['Single two-level']*len(db_sah)
dataDf=db_sah
dataDf['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf['Mode']]
exp.uploadDB(dataDf)
vocab={
    '0':'Two one-level',
    '1':'Single two-level',
}
# %%-

# %%--  Export data
exp.exportDataset()
# %%-
