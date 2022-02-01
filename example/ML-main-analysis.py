#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Initialization
#///////////////////////////////////////////
# %%--  Imports
import sys
# import the function file from another folder:
# sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\Yoann_code\DPML')
sys.path.append(r'C:\Users\budac\Documents\GitHub\Yoann_code\DPML')
from Si import *
from main import *
from utils import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
import os
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Set-up
#///////////////////////////////////////////
# %%--  Instructions:
'''
---Main Steps---
    Choose SAVEDIR folder where to save the output files from DPML. (Use absolute path)
    Provide TEMPERATURE as a list of the temperature in Kelvin for each measurements
    Provide DOPING as a list of the temperature in cm-3 for each measurements
    Provide cell type 'n' or 'p'
    NAME your experiment

---Other notes---
    Change hyper-parameters as desired.
    There are hidden parameters that can be specified in most functions, they
    use by the default the class-defined parameters
    Adjust ML pipeline to test for different ML algorithms. Don't forget to import the ML functions from Sklearn
'''
# %%-

# %%--  Inputs
SAVEDIR = r"C:\Users\budac\Documents\GitHub\Yoann_code\example\Savedir_example" # create a folder that save the output for DPML
TEMPERATURE = [200,250,300,350,400] # define a list of temperature for lifetime data generation (units are in K)
DOPING = [1e15,1e15,1e15,1e15,1e15] # define a list of doping levels for lifetime data generation (units are in cm3)
WAFERTYPE = 'p' # defien the doping type of the wafer for lifetime data generation
NAME = 'Main' # Name of the experiment.

#   File specific inputs
ML_REGRESSION_PIPELINE={
    "Random Forest": RandomForestRegressor(n_estimators=100, verbose =0, n_jobs=-1),
    "Adaptive Boosting": AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators=100, loss='linear'),
    "Gradient Boosting": GradientBoostingRegressor(verbose=0,loss='ls',max_depth=10),
    "Neural Network": MLPRegressor((100,100),alpha=0.001, activation = 'relu',verbose=0,learning_rate='adaptive'),
    "Support Vector": SVR(kernel='rbf',C=5,verbose=0, gamma="auto"),
}
# the hyper parameters for regression models
ML_CLASSIFICATION_PIPELINE={
    "Random Forest": RandomForestClassifier(n_estimators=100, verbose =0,n_jobs=-1),
    "Adaptive Boosting": AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), n_estimators=10),
    "Gradient Boosting": GradientBoostingClassifier(verbose=0,loss='deviance'),
    "Neural Network": MLPClassifier((100,100),alpha=0.001, activation = 'relu',verbose=0,learning_rate='adaptive'),
    "Nearest Neighbors":KNeighborsClassifier(n_neighbors = 5, weights='distance',n_jobs=-1),
}
# the hyper parameters for classification models
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

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Script
#///////////////////////////////////////////
# %%--  Define experiment and generate defect database
# os.path.exists(SAVEDIR)
exp = Experiment(SaveDir=SAVEDIR, Parameters=PARAMETERS) # define an experiment by defining its save direction and the experiment parameters.
exp.updateParameters({'type':WAFERTYPE,'temperature':TEMPERATURE,'doping':DOPING}) # set the wafer type, temperature, and the doping levels of the experiments.
exp.generateDB() # generate the lifetime data for the given experiment
# %%-
# %%--  Train machine learning algorithms loop: to into experiment file in DPML/main/experiment to see what these codes are doing
# prepare an empty list to collect r2 score.
# r2_list = []
for modelName,model in ML_REGRESSION_PIPELINE.items(): # for each maching learning model and corresponding hyper parameters, this for loop is doing regression only
    ml = exp.newML(mlParameters={'name':exp.parameters['name']+"_"+modelName}) # use newML method defined in class experiment to set a maching learning ID.
    # prepare an emtpy list to collect r2
    r2_list_model = []
    for trainKey in exp.parameters['regression_training_keys']:
        targetCol, bandgapParam = trainKey.rsplit('_',1)
        param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col'],'base_model':model}
        ml.trainRegressor(targetCol=targetCol, trainParameters=param)
        ml.plotRegressor(trainKey, plotParameters={'scatter_c':'black'})
        # r2_list_model.append(r2_score)
    # r2_list.append(r2_list)
for modelName,model in ML_CLASSIFICATION_PIPELINE.items(): # this for loop is doing classification.
    ml = exp.newML(mlParameters={'name':exp.parameters['name']+"_"+modelName})
    for trainKey in exp.parameters['classification_training_keys']:
        targetCol, bandgapParam = trainKey.rsplit('_',1)
        param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col'],'base_model':model}
        ml.trainClassifier(targetCol=targetCol, trainParameters=param)

# %%-

# %%--  Export data
exp.exportDataset()
exp.exportValidationset()
# %%-
