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

try: 
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except: 
    import tensorflow as tf
import numpy as np
try: 
    import cPickle
except: 
    import _pickle as cPickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from .djinn_fns import tree_to_nn_weights, tf_dropout_regression, \
               get_hyperparams, tf_continue_training



def load(model_name, model_path="./"):
    """Reload model and launch tensorflow session.

    Args:
        model_name (str): Name of model.
        model_path (str): Path to model.
        
    Returns: 
        Object: djinn regressor model.
    """
    try:
        with open("%s%s.pkl"%(model_path,model_name), "rb") as f:
            model=cPickle.load(f) 
        model.load_model(model_name, model_path)
        return(model)
    except: 
        print("Error loading model.")
        return() 


class DJINN_Regressor():
    """DJINN regression model.

    Args:
        n_trees (int): Number of trees in random forest 
                (equal to the number of neural networks).
        max_tree_depth (int): Maximum depth of decision tree. 
                       Neural network will have max_tree_depth-1 
                       hidden layers.
        dropout_keep_prob (float): Probability of keeping a neuron
                          in dropout layers.  
    """

    def __init__(self, n_trees=1, max_tree_depth=4, dropout_keep_prob=1.0):
        self.__n_trees = n_trees 
        self.__tree_max_depth = max_tree_depth
        self.__dropout_keep_prob = dropout_keep_prob
        self.__yscale = None
        self.__xscale = None
        self.__sess = None
        self.__regression = True
        self.modelname = None
        self.modelpath = None




    def get_hyperparameters(self, X, Y, weight_reg=1.0e-8, random_state=None):
        """Automatic selection of djinn hyper-parameters.
        
        Returns learning rate, number of epochs, batch size.
        
        Args: 
            X (ndarray): Input parameters for training data.
            Y (ndarray): Target parameters for training data. 
            weight_reg (float): Multiplier for L2 penalty on weights.
            random_state (int): Set the random seed. 

        Raises:
            Exception: if decision tree cannot be built from the data.

        Returns: 
            dictionary: Dictionary with batch size, 
                        learning rate, number of epochs
        """

        if (X.ndim == 1): 
            print('Please reshape single-input data to a one-column array')
            return


        single_output = False
        if (Y.ndim == 1): 
            single_output = True
            Y = Y.reshape(-1,1)

        # Scale the data
        self.__xscale = MinMaxScaler().fit(X)
        if self.__regression == True: 
            self.__yscale = MinMaxScaler().fit(Y)


        # Train the random forest
        rfr = RandomForestRegressor(self.__n_trees, max_depth=self.__tree_max_depth,
                                    bootstrap=True, random_state=random_state)
        if self.__regression == True: 
            if (single_output == True): 
                rfr.fit(self.__xscale.transform(X), self.__yscale.transform(Y).flatten())
            else: 
                rfr.fit(self.__xscale.transform(X), self.__yscale.transform(Y))

        else: 
            if (single_output == True): 
                rfr.fit(self.__xscale.transform(X), Y.flatten())
            else: 
                rfr.fit(self.__xscale.transform(X), Y)

        if(rfr.estimators_[0].tree_.max_depth < 1):
            raise Exception("Error: Cannot build decision tree.")

        # Map trees to initialized neural networks
        tree_to_network = tree_to_nn_weights(self.__regression, X, Y, self.__n_trees, rfr, random_state)

        print('Finding optimal hyper-parameters...')
        # Run auto-djinn 
        nn_batch_size, learnrate, nn_epochs = get_hyperparams(self.__regression, 
                tree_to_network, self.__xscale, self.__yscale, X, Y, 
                self.__dropout_keep_prob, weight_reg, random_state=random_state)       
 
        return({'batch_size':nn_batch_size, 'learn_rate':learnrate, 'epochs':nn_epochs})
        


    def train(self, X, Y, epochs=1000, learn_rate=0.001, batch_size=0, weight_reg=1.0e-8,
              display_step=1, save_files=True, file_name="djinn", save_model=True, 
              model_name="djinn_model", model_path="./", random_state=None):
        """Train djinn with specified hyperparameters.
        
        Args: 
            X (ndarray): Input parameters for training data.
            Y (ndarray): Target parameters for training data. 
            epochs (int): Number of training epochs.
            learn_rate (float): Learning rate for optimizaiton of weights/biases.
            batch_size (int): Number of samples per batch.
            weight_reg (float): Multiplier for L2 penalty on weights.
            display_step (int): Cost is printed every display_steps during training.
            save_files (bool): If True, saves train/valid cost per epoch, weights/biases.
            file_name (str): File name used if 'save_files' is True.
            save_model (bool): If True, saves the trained model.
            model_name (str): File name for model if 'save_model' is True.
            model_path (str): Location of where the model/files are saved. 
            random_state (int): Set the random seed. 

        Raises:
            Exception: if decision tree cannot be built from the data.
        
        Returns:
            None
        """
        self.modelname = model_name
        self.modelpath = model_path

        if (X.ndim == 1): 
            print('Please reshape single-input data to a one-column array')
            return

        # Reshape data to play well with sklearn
        single_output = False
        if (Y.ndim == 1): 
            single_output = True
            Y = Y.reshape(-1,1)

        # Create scalers 
        if(self.__xscale == None):
            self.__xscale = MinMaxScaler().fit(X)
            if self.__regression == True:
                self.__yscale = MinMaxScaler().fit(Y)

        # Train the random forest
        rfr = RandomForestRegressor(self.__n_trees, max_depth=self.__tree_max_depth,
                                    bootstrap=True, random_state=random_state)
        if self.__regression == True: 
            if (single_output == True): 
                rfr.fit(self.__xscale.transform(X), self.__yscale.transform(Y).flatten())
            else: 
                rfr.fit(self.__xscale.transform(X), self.__yscale.transform(Y))

        else: 
            if (single_output == True): 
                rfr.fit(self.__xscale.transform(X), Y.flatten())
            else: 
                rfr.fit(self.__xscale.transform(X), Y)

        # Check the forest was successful
        if(rfr.estimators_[0].tree_.max_depth <= 1):
            raise Exception("Error: Cannot build decision tree.")
        # Map trees to neural networks
        tree_to_network = tree_to_nn_weights(self.__regression, X, Y, self.__n_trees, rfr, random_state)
        print("DJINN Architecure: ", tree_to_network['network_shape']['tree_%s'%0])
        if(batch_size == 0): batch_size = int(np.ceil(0.05*len(Y)))
        tf_dropout_regression(self.__regression, tree_to_network, self.__xscale, 
                             self.__yscale, X, Y,ntrees=self.__n_trees,
                             filename=file_name, learnrate=learn_rate, 
                             training_epochs=epochs, batch_size=batch_size,
                             dropout_keep_prob=self.__dropout_keep_prob, weight_reg=weight_reg, 
                             display_step=display_step, savefiles=save_files,
                             savemodel=save_model, modelname=self.modelname,
                             modelpath=self.modelpath, random_state=random_state)
        if (save_model == True):
            with open('%s%s.pkl'%(self.modelpath, self.modelname), 'wb') as f:
                cPickle.dump(self, f)
         
        


    def fit(self, X, Y, epochs=None, learn_rate=None, batch_size=None, weight_reg=1.0e-8,  
            display_step=1, save_files=True, file_name="djinn", save_model=True,
            model_name="djinn_model", model_path="./", random_state=None):
        """Trains djinn model with optimal settings, if not supplied.

        Args: 
            X (ndarray): Input parameters for training data.
            Y (ndarray): Target parameters for training data. 
            epochs (int): Number of training epochs.
            learn_rate (float): Learning rate for optimizaiton of weights/biases.
            batch_size (int): Number of samples per batch.
            weight_reg (float): Multiplier for L2 penalty on weights.
            display_step (int): Cost is printed every display_steps during training.
            save_files (bool): If True, saves train/valid cost per epoch, weights/biases.
            file_name (str): File name used if 'save_files' is True.
            save_model (bool): If True, saves the trained model.
            model_name (str): File name for model if 'save_model' is True.
            model_path (str): Location of where the model/files are saved. 
            random_state (int): Set the random seed. 

        Returns:
            None
        """

        if(learn_rate == None):
            optimal=self.get_hyperparameters(X, Y, weight_reg, random_state)
            learn_rate=optimal['learn_rate']
            batch_size=optimal['batch_size']
            epochs=optimal['epochs']

        self.train(X, Y, epochs, learn_rate, batch_size, weight_reg,  
              display_step, save_files, file_name, save_model, 
              model_name, model_path, random_state)



    def load_model(self, model_name, model_path):
        """Reload tensorflow session for saved model. Called by djinn.load,

        Args: 
            model_path (str, optional): Location of model if different than 
                       location set during training.
            model_name (str, optional): Name of model if different than 
                       name set during training.
        
        Returns: 
            Object: djinn regressor model.
            
        """
        self.__sess = {}
        for p in range(0, self.__n_trees):
            tf.reset_default_graph()
            new_saver = \
            tf.train.import_meta_graph('%s%s_tree%s.ckpt.meta'%(model_path,model_name,p))
            self.__sess[p] = tf.Session()
            new_saver.restore(self.__sess[p], '%s%s_tree%s.ckpt'%(model_path,model_name,p))
            print("Model %s restored"%p)     
    


    def close_model(self):
        """Closes tensorflow sessions launched with djinn.load.
        
        Args:
            None
        Returns:
            None
        """
        for p in range(0, self.__n_trees):
            self.__sess[p].close()     
        


    def bayesian_predict(self, x_test, n_iters, random_state=None):
        """Bayesian distribution of predictions for a set of test data.

        Args:
            x_test (ndarray): Input parameters for test data.
            n_iters (int): Number of times to evaluate each neural network 
                           per test point.
            random_state (int): Set the random seed. 
       
        Returns:
            tuple (ndarray, ndarray, ndarray, dict):
                25th percentile of distribution of predictions for each test point.
                50th percentile of distribution of predictions for each test point.
                75th percentile of distribution of predictions for each test point.
                Dictionary containing inputs and predictions per tree, per
                  iteration, for each test point.
                  
        """    
        nonBayes = False
        if(n_iters == None):
            nonBayes = True
            n_iters = 1
        if(random_state): tf.set_random_seed(random_state)
        if(self.__sess == None): self.load_model(self.modelname, self.modelpath)
        if(x_test.ndim == 1): x_test = x_test.reshape(1,-1)
        samples = {}
        samples['inputs'] = x_test
        x_test = self.__xscale.transform(x_test)
        samples['predictions'] = {}
        for p in range(0, self.__n_trees):
            x = self.__sess[p].graph.get_tensor_by_name("input:0")
            keep_prob = self.__sess[p].graph.get_tensor_by_name("keep_prob:0")
            pred = self.__sess[p].graph.get_tensor_by_name("prediction:0")
            samples['predictions']['tree%s'%p] = \
                   [self.__yscale.inverse_transform(self.__sess[p].run(pred,\
                   feed_dict={x:x_test, keep_prob:self.__dropout_keep_prob})) 
                   for i in range(n_iters)]

        nout = samples['predictions']['tree0'][0].shape[1]
        preds = np.array([samples['predictions'][t] 
                for t in samples['predictions']]).reshape((n_iters*self.__n_trees, len(x_test), nout))

        middle = np.percentile(preds, 50, axis=0)
        lower = np.percentile(preds, 25, axis=0)
        upper = np.percentile(preds, 75, axis=0)
        if(nonBayes == True):
            return(np.mean(preds, axis=0))
        else:
            return(lower, middle, upper, samples)   

    
    def predict(self, x_test, random_state=None):
        """Predict target values for a set of test data.

        Args:
            x_test (ndarray): Input parameters for test data.
            random_state (int): Set the random seed. 
       
        Returns:
            ndarray: Mean target value prediction for each test point.
        """ 
        return self.bayesian_predict(x_test, None, random_state)
    
    
    def collect_tree_predictions(self, predictions):
        """Gather distributions of predictions for each test point.

        Args: 
            predictions (dict): The 'predictions' key from the dictionary 
                        returned by bayesian_predict.

        Returns:
            ndarray: Re-shaped predictions (niters*ntrees, # test points, output dim) 
        """
        nout = predictions['tree0'][0].shape[1]
        n_iters = len(predictions['tree0'])
        xlength = predictions['tree0'][0].shape[0]
        preds = np.array([predictions[t] 
                for t in predictions]).reshape((n_iters*self.__n_trees, xlength, nout))
        return(preds)
      

    def continue_training(self, X, Y, training_epochs, learn_rate, batch_size, display_step=1, random_state=None):
        """Continue training an exisitng model. Must load_model first.
        
        Model is resaved in current location.

        Args: 
            X (ndarray): Input parameters for training data.
            Y (ndarray): Target parameters for training data. 
            epochs (int): Number of training epochs.
            learn_rate (float): Learning rate for optimizaiton of weights/biases.
            batch_size (int): Number of samples per batch.
            display_step (int): Cost is printed every display_steps during training.
            random_state (int): Set the random seed. 

        Returns:
            None
        """
        ntrees=self.__n_trees
        nhl=self.__tree_max_depth-1
        dropout_keep_prob=self.__dropout_keep_prob
        tf_continue_training(self.__regression, self.__xscale, self.__yscale, 
                            X, Y, ntrees, learn_rate, training_epochs, batch_size,
                            self.__dropout_keep_prob, nhl, display_step,
                            self.modelname, self.modelpath, random_state)




class DJINN_Classifier(DJINN_Regressor):
    """DJINN classification model.

    Args:
        n_trees (int): Number of trees in random forest 
                (equal to the number of neural networks).
        max_tree_depth (int): Maximum depth of decision tree. 
                       Neural network will have max_tree_depth-1 
                       hidden layers.
        dropout_keep_prob (float): Probability of keeping a neuron
                          in dropout layers.  
    """
    def __init__(self, n_trees=1, max_tree_depth=4, dropout_keep_prob=1.0):
        DJINN_Regressor.__init__(self, n_trees, max_tree_depth, dropout_keep_prob)
        self._DJINN_Regressor__regression = False

    def bayesian_predict(self, x_test, n_iters, random_state=None):
        """Bayesian distribution of predictions for a set of test data.

        Args:
            x_test (ndarray): Input parameters for test data.
            n_iters (int): Number of times to evaluate each neural network 
                           per test point.
            random_state (int): Set the random seed. 
       
        Returns:
            tuple (ndarray, ndarray, ndarray, dict):
                25th percentile of distribution of predictions for each test point.
                50th percentile of distribution of predictions for each test point.
                75th percentile of distribution of predictions for each test point.
                Dictionary containing inputs and predictions per tree, per
                  iteration, for each test point.
                  
        """    
        nonBayes = False
        if(n_iters == None):
            nonBayes = True
            n_iters = 1
        if(random_state): tf.set_random_seed(random_state)
        if(self._DJINN_Regressor__sess == None): self.load_model(self.modelname, self.modelpath)
        if(x_test.ndim == 1): x_test = x_test.reshape(1,-1)
        samples = {}
        samples['inputs'] = x_test
        x_test = self._DJINN_Regressor__xscale.transform(x_test)
        samples['predictions'] = {}
        for p in range(0, self._DJINN_Regressor__n_trees):
            x = self._DJINN_Regressor__sess[p].graph.get_tensor_by_name("input:0")
            keep_prob = self._DJINN_Regressor__sess[p].graph.get_tensor_by_name("keep_prob:0")
            pred = self._DJINN_Regressor__sess[p].graph.get_tensor_by_name("prediction:0")
            samples['predictions']['tree%s'%p] = \
                  [self._DJINN_Regressor__sess[p].run(pred,\
                  feed_dict={x:x_test, keep_prob:self._DJINN_Regressor__dropout_keep_prob}) for i in range(n_iters)]

        nout = samples['predictions']['tree0'][0].shape[1]
        preds = np.array([samples['predictions'][t] 
                for t in samples['predictions']]).reshape((n_iters*self._DJINN_Regressor__n_trees, len(x_test), nout))
        print(preds.shape)
        middle = np.argmax(np.percentile(preds,50,axis=0),1)
        lower = np.argmax(np.percentile(preds,25,axis=0), 1)
        upper = np.argmax(np.percentile(preds,75,axis=0), 1)

        if(nonBayes == True):
            return(middle)
        else:
            return(lower, middle, upper, samples)

    
