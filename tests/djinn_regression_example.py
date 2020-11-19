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
# Boston housing dataset. Please see comments and djinn docs for
# details on each function. 
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import sklearn
try: from sklearn.model_selection import train_test_split
except: from sklearn.cross_validation import train_test_split

from sklearn import datasets
from djinn import djinn
print(sklearn.__version__)

''' 
    NOTE: for the boston housing data you can expect test 
    MSE~10-20, Mean Abs Err~3-4, Exp.Var.~0.8+
    when using get_hyperparameters() function
'''

#Load the data, split into training/testing groups
d=datasets.load_boston()
X=d.data
Y=d.target

x_train,x_test,y_train,y_test=train_test_split(X, Y, test_size=0.2, random_state=1) 

print("djinn example")    
modelname="reg_djinn_test"   # name the model
ntrees=1                 # number of trees = number of neural nets in ensemble
maxdepth=4               # max depth of tree -- optimize this for each data set
dropout_keep=1.0         # dropout typically set to 1 for non-Bayesian models

#initialize the model
model=djinn.DJINN_Regressor(ntrees,maxdepth,dropout_keep)

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
mse=sklearn.metrics.mean_squared_error(y_test,m)
mabs=sklearn.metrics.mean_absolute_error(y_test,m)
exvar=sklearn.metrics.explained_variance_score(y_test,m)   
print('MSE',mse)
print('M Abs Err',mabs)
print('Expl. Var.',exvar)

#close model 
model.close_model()

print("Reload model and continue training for 20 epochs")
# reload model; can also open it using cPickle.load()
model2=djinn.load(model_name="reg_djinn_test")

#continue training for 20 epochs using same learning rate, etc as before
model2.continue_training(x_train, y_train, 20, learnrate, batchsize, random_state=1)

#make updated predictions
m2=model2.predict(x_test)

#evaluate results
mse2=sklearn.metrics.mean_squared_error(y_test,m2)
mabs2=sklearn.metrics.mean_absolute_error(y_test,m2)
exvar2=sklearn.metrics.explained_variance_score(y_test,m2)   
print('MSE',mse2)
print('M Abs Err',mabs2)
print('Expl. Var.',exvar2)


# Bayesian formulation with dropout. Recommend dropout keep 
# probability ~0.95, 5-10 trees.
print("Bayesian djinn example")
ntrees=3
dropout_keep=0.95
modelname="reg_bdjinn_test"

# initialize a model
bmodel=djinn.DJINN_Regressor(ntrees,maxdepth,dropout_keep)

# "fit()" does what get_hyperparameters + train does, in one step: 
bmodel.fit(x_train,y_train, display_step=1, save_files=True, file_name=modelname, 
           save_model=True,model_name=modelname, random_state=1)

# evaluate: niters is the number of times you evaluate the network for 
# a single sample. higher niters = better resolved distribution of predictions
niters=100
bl,bm,bu,results=bmodel.bayesian_predict(x_test,n_iters=niters, random_state=1)
# bayesian_predict returns 25, 50, 75 percentile and results dict with all predictions

# evaluate performance on median predictions
mse=sklearn.metrics.mean_squared_error(y_test,bm)
mabs=sklearn.metrics.mean_absolute_error(y_test,bm)
exvar=sklearn.metrics.explained_variance_score(y_test,bm)   
print('MSE',mse)
print('M Abs Err',mabs)
print('Expl. Var.',exvar)

# make a pretty plot
yerrors = np.column_stack((bm-bl, bu-bm)).reshape((2,bl.shape[0]))
g=np.linspace(np.min(y_test),np.max(y_test),10)    
fig, axs = plt.subplots(1,1, figsize=(8,8), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .15, wspace=.1)
sc=axs.scatter(y_test, bm, linewidth=0,s=6, 
                  alpha=0.8, c='#68d1ca')
a,b,c=axs.errorbar(y_test, bm, yerr=yerrors, marker='',ls='',zorder=0, 
                   alpha=0.5, ecolor='black')
axs.set_xlabel("True")
axs.set_ylabel("B-DJINN Prediction")    
axs.plot(g,g,color='red')
plt.show()

# collect_tree_predictions gathers predictions in results dict
# in a more intuitive way for easy plotting, etc
p=bmodel.collect_tree_predictions(results['predictions'])


