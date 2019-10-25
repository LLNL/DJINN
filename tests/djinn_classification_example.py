###############################################################################
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
#
# Written by K. Humbird (humbird1@llnl.gov), L. Peterson (peterson76@llnl.gov).
#
# LLNL-CODE-754815
#
# All rights reserved.
#
# This file is part of DJINN.
#
# For details, see github.com/LLNL/djinn. 
#
# For details about use and distribution, please read DJINN/LICENSE .
###############################################################################

###############################################################################
# Demo script for DJINN 
# Below, each function available in DJINN is demonstrated for the 
# iris classification dataset. Please see comments and djinn docs for
# details on each function. 
###############################################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
try: from sklearn.model_selection import train_test_split
except: from sklearn.cross_validation import train_test_split

from sklearn import datasets
from djinn import djinn
print(sklearn.__version__)


#Load the data, split into training/testing groups
d=datasets.load_iris()
X=d.data
Y=d.target

x_train,x_test,y_train,y_test=train_test_split(X, Y, test_size=0.2, random_state=1) 

print("djinn example")    
modelname="class_djinn_test"   # name the model
ntrees=1                 # number of trees = number of neural nets in ensemble
maxdepth=4               # max depth of tree -- optimize this for each data set
dropout_keep=1.0         # dropout typically set to 1 for non-Bayesian models

#initialize the model
model=djinn.DJINN_Classifier(ntrees,maxdepth,dropout_keep)

# find optimal settings: this function returns dict with hyper-parameters
# each djinn function accepts random seeds for reproducible behavior
optimal=model.get_hyperparameters(x_train, y_train, random_state=1)
batchsize=optimal['batch_size']
learnrate=optimal['learn_rate']
epochs=optimal['epochs']


# train the model with hyperparameters determined above
model.train(x_train,y_train,epochs=epochs,learn_rate=learnrate, batch_size=batchsize, 
              display_step=1, save_files=True, file_name=modelname, 
              save_model=True,model_name=modelname, random_state=1)

# *note there is a function model.fit(x_train,y_train, ... ) that wraps 
# get_hyperparameters() and train(), so that you do not have to manually
# pass hyperparameters to train(). However, get_hyperparameters() can
# be expensive, so I recommend running it once per dataset and using those
# hyperparameter values in train() to save computational time

# make predictions
m=model.predict(x_test) #returns the median prediction if more than one tree

#evaluate results
acc=sklearn.metrics.accuracy_score(y_test,m.flatten())  
print('Accuracy',acc)

#close model 
model.close_model()

print("Reload model and continue training")
# reload model; can also open it using cPickle.load()
model2=djinn.load(model_name="class_djinn_test")

#continue training for 20 epochs using same learning rate, etc as before
model2.continue_training(x_train, y_train, 20, learnrate, batchsize, random_state=1)

#make updated predictions
m2=model2.predict(x_test)

#evaluate results
acc=sklearn.metrics.accuracy_score(y_test,m.flatten())  
print('Accuracy',acc)


# Bayesian formulation with dropout. Recommend dropout keep 
# probability ~0.95, 5-10 trees.
print("Bayesian djinn example")
ntrees=3
dropout_keep=0.95
modelname="class_bdjinn_test"

# initialize a model
bmodel=djinn.DJINN_Classifier(ntrees,maxdepth,dropout_keep)

# "fit()" does what get_hyperparameters + train does, in one step: 
bmodel.fit(x_train,y_train, display_step=1, save_files=True, file_name=modelname, 
           save_model=True,model_name=modelname, random_state=1)

# evaluate: niters is the number of times you evaluate the network for 
# a single sample. higher niters = better resolved distribution of predictions
niters=100
bl,bm,bu,results=bmodel.bayesian_predict(x_test,n_iters=niters, random_state=1)
# bayesian_predict returns 25, 50, 75 percentile and results dict with all predictions

# evaluate performance on median predictions
acc=sklearn.metrics.accuracy_score(y_test,bm.flatten())  
print('Accuracy',acc)

# collect_tree_predictions gathers predictions in results dict
# in a more intuitive way for easy plotting, etc
p=bmodel.collect_tree_predictions(results['predictions'])


