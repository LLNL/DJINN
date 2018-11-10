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
# Multiple-output regression demo script for DJINN 
# Please see comments and djinn docs for details. 
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from djinn import djinn
try: from sklearn.model_selection import train_test_split
except: from sklearn.cross_validation import train_test_split



# For the boston housing data you can expect final 
# test MSE~10-20, Mean Abs Err~3-4, Exp.Var.~0.8+
# when using get_hyperparameters() 


#Load the data, split into training/testing groups
d=sklearn.datasets.load_boston()
X=d.data
Y=d.target
Y=np.column_stack((Y,0.5*Y))   # make two columns of outputs
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2) 
   
print("Create DJINN model with multiple outputs")
modelname="multireg_djinn_test"    # name the model
ntrees=1                        # number of trees = number of neural nets in ensemble
maxdepth=4                      # max depth of tree -- optimize this for each data set
dropout_keep=1.0                # dropout typically set to 1 for non-Bayesian models

# initialize the model
model=djinn.DJINN_Regressor(ntrees,maxdepth,dropout_keep)

# find optimal settings
optimal=model.get_hyperparameters(x_train,y_train)
batchsize=optimal['batch_size']
learnrate=optimal['learn_rate']
epochs=np.min((300,optimal['epochs']))

# train the model with these settings
model.train(x_train,y_train, epochs=epochs,learn_rate=learnrate, batch_size=batchsize, 
              display_step=1, save_files=True, file_name=modelname, 
              save_model=True,model_name=modelname)

m=model.predict(x_test)

# evaluate results
for i in [0,1]:
    mse=sklearn.metrics.mean_squared_error(y_test[:,i],m[:,i])
    mabs=sklearn.metrics.mean_absolute_error(y_test[:,i],m[:,i])
    exvar=sklearn.metrics.explained_variance_score(y_test[:,i],m[:,i])   
    print('MSE',mse)
    print('M Abs Err',mabs)
    print('Expl. Var.',exvar)


# close model 
model.close_model()

print("Reload model and continue training for 50 epochs")

# reload model and continue training for 50 more epochs
model2=djinn.load(model_name="multireg_djinn_test")

model2.continue_training(x_train, y_train, 50, learnrate, batchsize)

m2=model2.predict(x_test)

# evaluate results
mse2=sklearn.metrics.mean_squared_error(y_test,m2)
mabs2=sklearn.metrics.mean_absolute_error(y_test,m2)
exvar2=sklearn.metrics.explained_variance_score(y_test,m2)   
print('MSE',mse2)
print('M Abs Err',mabs2)
print('Expl. Var.',exvar2)




print ("Create Bayesian-DJINN model with multiple outputs")
modelname="multireg_bdjinn_test"  #name the model 
ntrees=3                       # make ensemble of (3) models
dropout_keep=0.95              # turn on dropout for error bars

# initilaize model and get hyper-parameters
bmodel=djinn.DJINN_Regressor(ntrees,maxdepth,dropout_keep)
optimal=bmodel.get_hyperparameters(x_train,y_train)
batchsize=optimal['batch_size']
learnrate=optimal['learn_rate']
epochs=optimal['epochs']

# train the model with these settings
bmodel.train(x_train,y_train,epochs=epochs,learn_rate=learnrate,batch_size=batchsize, 
             display_step=1, save_files=True, file_name=modelname, 
              save_model=True,model_name=modelname)

# evaluate 
niters=50
bl,bm,bu,results=bmodel.bayesian_predict(x_test,n_iters=niters)

for i in [0,1]:
    mse=sklearn.metrics.mean_squared_error(y_test[:,i],m[:,i])
    mabs=sklearn.metrics.mean_absolute_error(y_test[:,i],m[:,i])
    exvar=sklearn.metrics.explained_variance_score(y_test[:,i],m[:,i])   
    print('MSE',mse)
    print('M Abs Err',mabs)
    print('Expl. Var.',exvar)

#Make a pretty plot
g=np.linspace(np.min(y_test[:,0]),np.max(y_test[:,0]),10)    
fig, axs = plt.subplots(1,1, figsize=(8,8), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .15, wspace=.1)
sc=axs.scatter(y_test[:,0], bm[:,0], linewidth=0,s=6, 
                  alpha=0.8, c='#68d1ca')
a,b,c=axs.errorbar(y_test[:,0], bm[:,0], yerr=[bm[:,0]-bl[:,0],bu[:,0]-bm[:,0]], marker='',ls='',zorder=0, 
                   alpha=0.5, ecolor='black')
axs.set_xlabel("True")
axs.set_ylabel("B-DJINN Prediction")    
axs.plot(g,g,color='red')



#Make a pretty plot
g=np.linspace(np.min(y_test[:,1]),np.max(y_test[:,1]),10)    
fig, axs = plt.subplots(1,1, figsize=(8,8), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .15, wspace=.1)
sc=axs.scatter(y_test[:,1], bm[:,1], linewidth=0,s=6, 
                  alpha=0.8, c='#68d1ca')
a,b,c=axs.errorbar(y_test[:,1], bm[:,1], yerr=[bm[:,1]-bl[:,1],bu[:,1]-bm[:,1]], marker='',ls='',zorder=0, 
                   alpha=0.5, ecolor='black')
axs.set_xlabel("True")
axs.set_ylabel("B-DJINN Prediction")    
axs.plot(g,g,color='red')
plt.show()

print("test collect tree predictions fn")
p=bmodel.collect_tree_predictions(results['predictions'])
print(p.shape)

