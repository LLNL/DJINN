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
from sklearn.tree import _tree
from sklearn.preprocessing import MinMaxScaler
try: from sklearn.model_selection import train_test_split
except: from sklearn.cross_validation import train_test_split
try: import cPickle
except: import _pickle as cPickle
        
        
def tree_to_nn_weights(regression, X, Y, num_trees, rfr, random_state) :
    """ Main function to map tree to neural network. Determines architecture, initial weights.  
        
    Args:
        x (ndarray): Input features.
        y (ndarray): Output features.
        ntrees (int): Number of decision trees.
        rfr (object): Random forest regressor.
        random_state (int): Sets random seed.

    Returns:
        dict: includes weights, biases, architecture
    """

    def xav(nin,nout):
        """Xavier initialization. args: input & output dim of layer """
        return(np.random.normal(loc=0.0,scale=np.sqrt(3.0/(nin+nout))))

    #set seed
    if random_state: np.random.seed(random_state)

    #get input & output dimensions from data
    nin = X.shape[1]
    if regression == True:
        if(Y.size > Y.shape[0]): nout = Y.shape[1]
        else: nout = 1

    else: 
        nout=len(np.unique(Y))

    #store nn info to pass to tf
    tree_to_network={}
    tree_to_network['n_in'] = nin
    tree_to_network['n_out'] = nout
    tree_to_network['network_shape']={}
    tree_to_network['weights']={}
    tree_to_network['biases']={}

    #map each tree in RF to initial weights
    for tree in range(num_trees):
        #pull data from tree
        tree_ = rfr.estimators_[tree].tree_
        features=tree_.feature
        n_nodes = tree_.node_count
        children_left = tree_.children_left
        children_right = tree_.children_right
        threshold = tree_.threshold
        
        # Traverse tree structure to record node depth, node id, etc
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        
        
        # collect tree structure in dict, sorted by node number
        node={}
        for i in range(len(features)):
            node[i]={}
            node[i]['depth']=node_depth[i]
            if(features[i]>=0): node[i]['feature']=features[i]
            else: node[i]['feature']=-2
            node[i]['child_left']=features[children_left[i]]
            node[i]['child_right']=features[children_right[i]]
            
            
        # meta data arrays for mapping to djinn weights
        num_layers=len(np.unique(node_depth))  #number of layers in nn  
        nodes_per_depth=np.zeros(num_layers)   #number nodes in each layer of tree
        leaves_per_depth=np.zeros(num_layers)  #number leaves in each layer of tree  
        
        for i in range(num_layers):
            ind=np.where(node_depth==i)[0]
            nodes_per_depth[i]=len(np.where(features[ind]>=0)[0])
            leaves_per_depth[i]=len(np.where(features[ind]<0)[0])    
    
        #max depth at which each feature appears in tree
        max_depth_feature=np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            ind=np.where(features==i)[0]
            if(len(ind)>0): max_depth_feature[i]=np.max(node_depth[ind])
        
        #djinn architecture    
        djinn_arch=np.zeros(num_layers, dtype='int')  
    
        #hidden layers = previous layer + # new nodes in layer of tree 
        djinn_arch[0]=nin
        for i in range(1,num_layers):
            djinn_arch[i]=djinn_arch[i-1]+nodes_per_depth[i]
        djinn_arch[-1]=nout
    
        #create dict for djinn weights : create blank arrays
        djinn_weights={}
        for i in range(num_layers-1):
            djinn_weights[i]=np.zeros((djinn_arch[i+1],djinn_arch[i]))
        
        #create list of indices for new neurons in each layer    
        new_n_ind=[]
        for i in range(num_layers-1):
            new_n_ind.append(np.arange(djinn_arch[i],djinn_arch[i+1]))  
            
        #fill in weights in djinn arrays
        for i in range(num_layers-1): #loop through layers
            nn_in = djinn_weights[i].shape[0]
            nn_out = djinn_weights[i].shape[1]
            for f in range(nin): 
                #add diagonal terms up to depth feature is used
                if(i < max_depth_feature[f]-1): djinn_weights[i][f,f]=1.0
            #begin mapping off diagonal connections
            k=0; kk=0; #k keeps track of outgoing layer, kk keeps track of incoming layer neuron index
            for nodes in node:
                if node[nodes]['depth']== i :
                    feature=node[nodes]['feature']
                    if feature >= 0: #if node is a split/not a leaf
                        left=node[nodes]['child_left']
                        right=node[nodes]['child_right']
                        if((nodes==0) and ((left<0) or (right<0)) ): 
                            #leaf at first split: connect through out layer
                            for j in range(i,num_layers-2): djinn_weights[j][feature,feature]=1.0 
                            djinn_weights[num_layers-2][:,feature]=1.0
                        if(left>=0): 
                            #left child is split, connect nodes in that decision 
                            #to new neuron in current nn layer
                            if(i==0): djinn_weights[i][new_n_ind[i][k],feature] = xav(nn_in,nn_out)
                            else: djinn_weights[i][new_n_ind[i][k],new_n_ind[i-1][kk]] = xav(nn_in,nn_out)
                            djinn_weights[i][new_n_ind[i][k],left] = xav(nn_in,nn_out)
                            k+=1
                            if( kk >= len(new_n_ind[i-1]) ): kk=0 
                        if( (left<0) and (nodes!=0) ): #leaf- connect through to outputs
                            lind=new_n_ind[i-1][kk]
                            for j in range(i,num_layers-2): djinn_weights[j][lind,lind]=1.0 
                            djinn_weights[num_layers-2][:,lind]=1.0
                        if(right>=0):
                            #right child is split, connect nodes in that decision 
                            #to new neuron in current nn layer
                            if(i==0): djinn_weights[i][new_n_ind[i][k],feature]=xav(nn_in,nn_out)
                            else: djinn_weights[i][new_n_ind[i][k],new_n_ind[i-1][kk]]=xav(nn_in,nn_out)
                            djinn_weights[i][new_n_ind[i][k],right]=xav(nn_in,nn_out)
                            k+=1
                            if( kk >= len(new_n_ind[i-1])): kk=0 
                        if( (right<0) and (nodes!=0) ): #leaf- connect through to outputs
                            lind=new_n_ind[i-1][kk]
                            for j in range(i,num_layers-2): djinn_weights[j][lind,lind]=1.0 
                            djinn_weights[num_layers-2][:,lind]=1.0
                        kk+=1    
    
        #connect active neurons to output layer                         
        m=len(new_n_ind[-2])
        ind=np.where(abs(djinn_weights[num_layers-3][:,-m:])>0)[0]
        for inds in range(len(djinn_weights[num_layers-2][:,ind])):
            djinn_weights[num_layers-2][inds,ind]=xav(nn_in,nn_out)
    
        # dump weights, arch, biases into dict to pass to tf
        tree_to_network['network_shape']['tree_%s'%tree] = djinn_arch
        tree_to_network['weights']['tree_%s'%tree] = djinn_weights
        tree_to_network['biases']['tree_%s'%tree] = [] #maybe add biases
    return tree_to_network
 
    


def tf_dropout_regression(regression, ttn, xscale, yscale, x1, y1, ntrees, filename, 
                          learnrate, training_epochs, batch_size,
                          dropout_keep_prob, weight_reg, display_step, savefiles, 
                          savemodel, modelname, modelpath, random_state):
    """ Trains neural networks in tensorflow, given initial weights from decision tree.
        
    Args:
        ttn (dict): Dictionary returned from function tree_to_nn_weights.
        xscale (object): Input scaler.
        yscale (object): Output scaler.
        x1 (ndarray): Input features.
        y1 (ndarray): Output features.
        ntrees (int): Number of decision trees.
        filename (str): Name for saved files.
        learn_rate (float): Learning rate for optimizaiton of weights/biases.
        training_epochs (int): Number of epochs to train neural network.
        batch_size (int): Number of samples per batch.
        dropout_keep_prob (float): Probability of keeping neuron "on" in dropout layers.
        weight_reg (float): Multiplier for L2 penalty on weights.
        display_step (int): Cost is printed every display_steps during training.
        save_files (bool): If True, saves train/valid cost per epoch, weights/biases.
        file_name (str): File name used if 'save_files' is True.
        save_model (bool): If True, saves the trained model.
        model_name (str): File name for model if 'save_model' is True.
        model_path (str): Location of where the model/files are saved. 
        random_state (int): Sets random seed.

    Returns:
        dict: final neural network info: weights, biases, cost per epoch.
    """
    #get size of input/output layer
    n_input = ttn['n_in']    
    n_classes = ttn['n_out']
    #save min/max values for python-only djinn eval
    input_min = np.min(x1, axis=0)
    input_max = np.max(x1, axis=0)
    if(n_classes == 1): y1 = y1.reshape(-1,1)
    output_min = np.min(y1, axis=0)
    output_max = np.max(y1, axis=0)

    #scale data
    x1 = xscale.transform(x1)
    if regression == True:
        if(n_classes == 1): y1 = yscale.transform(y1).flatten()
        else: y1 = yscale.transform(y1)

    xtrain, xtest, ytrain, ytest = train_test_split(x1, y1, test_size=0.1, random_state=random_state) 

    #for classification, do one-hot encoding on classes
    if regression == False:
        ytrain=np.equal.outer(ytrain.flatten(), np.arange(len(np.unique(ytrain)))).astype("float32")
        ytest=np.equal.outer(ytest.flatten(), np.arange(len(np.unique(ytest)))).astype("float32")

    pp = 0
    #create dict/arrays to save network info
    nninfo = {}
    nninfo['input_minmax'] = [input_min, input_max]
    nninfo['output_minmax'] = [output_min, output_max]
    nninfo['initial_weights'] = {}; nninfo['initial_biases'] = {}; 
    nninfo['final_weights'] = {}; nninfo['final_biases'] = {}; 
    train_accur = np.zeros((ntrees, training_epochs))
    valid_accur = np.zeros((ntrees, training_epochs))
    #loop through trees, training each network in ensemble
    for keys in ttn["weights"]:
        tf.reset_default_graph()
        if (random_state): tf.set_random_seed(random_state) 

        #make placeholders for inputs/outputs/dropout
        x = tf.placeholder("float32", [None, n_input], name="input")
        if(n_classes == 1): y = tf.placeholder("float32", [None], name="target")
        else: y = tf.placeholder("float32", [None, n_classes], name="target")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        #get network shape from djinn mapping
        npl = ttn['network_shape'][keys]
        nhl = len(npl)-2
        n_hidden = {}
        for i in range(1, len(npl)-1):
            n_hidden[i] = npl[i]

        #create fully connected MLP
        def multilayer_perceptron(x, weights, biases):
            layer = {}
            layer[1] = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
            layer[1] = tf.nn.relu(layer[1])
            for i in range(2, nhl+1):
                layer[i] = tf.add(tf.matmul(layer[i-1], weights['h%s'%i]), biases['h%s'%i])
                layer[i] = tf.nn.relu(layer[i])   
                layer[i] = tf.nn.dropout(layer[i], keep_prob)
            out_layer = tf.matmul(layer[nhl], weights['out'])
            out_final = tf.add(out_layer, biases['out'], name="prediction")
            return out_final

        #get weights from djinn mapping; biases are random
        w = {}; b = {};
        for i in range(0, len(ttn['network_shape'][keys])-1):
            w[i+1] = np.transpose(ttn['weights'][keys][i]).astype((np.float32))

        weights = {}
        for i in range(1, nhl+1):
            weights['h%s'%i] = tf.Variable(w[i], name="w%s"%i)
        weights['out'] = tf.Variable(w[nhl+1], name="w%s"%(nhl+1))  
        biases = {}
        for i in range(1, nhl+1):
            biases['h%s'%i] = tf.Variable(tf.random_normal([int(n_hidden[i])], mean=0.0, 
                                          stddev=np.sqrt(3.0/(n_classes+int(n_hidden[nhl])))), name="b%s"%i)
        biases['out'] = tf.Variable(tf.random_normal([n_classes], mean=0.0, 
                                    stddev=np.sqrt(3.0/(n_classes+int(n_hidden[nhl])))), name="b%s"%nhl)  
        
        #prediction is the output from the MLP
        pred = multilayer_perceptron(x, weights, biases)
        if(n_classes == 1): predictions = tf.reshape(pred, [-1])
        else: predictions = pred     

        #for classification, we need to calculate the accuracy
        if regression == False:             
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), "float32"))

        #l2 weight penalty + mse cost fn  
        weight_decay = tf.reduce_sum(weight_reg * tf.stack([tf.nn.l2_loss(weights['h%s'%i]) 
                                         for i in range(1,nhl+1)] ))
        if regression == False:
            cost = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)), 
                          weight_decay, name="cost") 
        else:
            cost = tf.add(tf.reduce_mean(tf.square(y-predictions)), 
                      weight_decay, name="cost") 

        #use adam optimizer from tflearn                                          
        #optimize = tf.contrib.layers.optimize_loss(loss=cost,
        #            global_step=tf.train.get_global_step(),
        #            learning_rate=learnrate, optimizer="Adam")
        #optimizer=tf.add(optimize,0,name="opt")

        optimizer = tf.train.AdamOptimizer(learning_rate=learnrate).minimize(loss=cost,  global_step=tf.train.get_global_step(),name="Adam")

        #initialize vars & launch session
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            #save init weights/biases
            nninfo['initial_weights'][keys] = sess.run(weights)
            nninfo['initial_biases'][keys] = sess.run(biases)
            #train in minibatches
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(len(xtrain)/float(batch_size))
                for i in range(total_batch):
                    indx = np.random.randint(len(xtrain), size=batch_size)
                    batch_x, batch_y = xtrain[indx,:], ytrain[indx]
                    _, c, p = sess.run([optimizer, cost, pred], 
                                        feed_dict={x: batch_x, y: batch_y, 
                                        keep_prob:dropout_keep_prob})
                    avg_cost += c / total_batch
                train_accur[pp][epoch] = avg_cost
                valid_accur[pp][epoch] = cost.eval({x: xtest, y: ytest, keep_prob:dropout_keep_prob}) 
                # display training progresss
                if epoch % display_step == 0:
                    if regression == True:
                        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                    #for classification, print cost and accuracy:
                    else: 
                        avg_accuracy = accuracy.eval({x: xtrain, y: ytrain, keep_prob:dropout_keep_prob})
                        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost),
                               "accuracy=", "{:.3f}".format(avg_accuracy))
            print("Optimization Finished!")

            #save final weights/biases
            nninfo['final_weights'][keys] = sess.run(weights)
            nninfo['final_biases'][keys] = sess.run(biases)
            #save model
            if(savemodel == True):
                save_path = saver.save(sess, "%s%s_tree%s.ckpt"%(modelpath, modelname, pp))
                print("Model saved in: %s" % save_path)
        sess.close()    
        pp += 1 
   
    #save files with nn info  
    nninfo['train_cost'] = train_accur
    nninfo['valid_cost'] = valid_accur
    if(savefiles == True):        
        with open('%snn_info_%s.pkl'%(modelpath, filename), 'wb') as f1:
            cPickle.dump(nninfo, f1)
    return(nninfo, npl)


def get_hyperparams(regression, ttn, xscale, yscale, x1, y1, dropout_keep_prob, 
                    weight_reg, random_state):
    """Performs search for automatic selection of djinn hyper-parameters.
        
    Returns learning rate, number of epochs, batch size.
        
    Args: 
        ttn (dict): Dictionary returned from function tree_to_nn_weights.
        xscale (object): Input scaler.
        yscale (object): Output scaler.
        x1 (ndarray): Input features.
        y1 (ndarray): Output features.
        dropout_keep_prob (float): Probability of keeping neuron "on" in dropout layers.
        weight_reg (float): Multiplier for L2 penalty on weights.
        random_state (int): Sets random seed.

        Returns: 
            dictionary: Dictionary with batch size, 
                        learning rate, number of epochs
        """
    #get size of input/output
    n_input = ttn['n_in']    
    n_classes = ttn['n_out']

    #scale data
    x1 = xscale.transform(x1)
    if regression == True: 
        if(n_classes == 1): y1 = yscale.transform(y1).flatten()
        else: y1 = yscale.transform(y1)  


    xtrain, xtest, ytrain, ytest = train_test_split(x1, y1, test_size=0.1, random_state=random_state) 

    #for classification, do one-hot encoding on classes
    if regression == False:
        ytrain=np.equal.outer(ytrain.flatten(), np.arange(len(np.unique(ytrain)))).astype("float32")
        ytest=np.equal.outer(ytest.flatten(), np.arange(len(np.unique(ytest)))).astype("float32")

    ystar = {}
    ystar['preds'] = {}

    #learning rates to test
    minlr = -4.0
    maxlr = -2.0
    lrs = np.logspace(minlr, maxlr, 10)

    #default batch size 
    batch_size = int(np.ceil(0.05*len(y1)))

    #optimizing on one tree only
    keys = 'tree_0'
    print("Determining learning rate...")
    for lriter in range(0, 2): #iterate twice
        #arrays for performance data
        accur = np.zeros((len(lrs), 100))
        errormin = np.zeros(len(lrs))
        for pp in range(0, len(lrs)):
            tf.reset_default_graph()
            if (random_state): tf.set_random_seed(random_state)

            x = tf.placeholder("float32",[None, n_input], name="input")
            if(n_classes == 1): y = tf.placeholder("float32",[None])
            else: y = tf.placeholder("float32",[None, n_classes])
            keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            npl = ttn['network_shape'][keys]
            nhl = len(npl)-2
            n_hidden = {}
            for i in range(1, len(npl)-1):
                n_hidden[i] = npl[i]

            def multilayer_perceptron(x, weights, biases):
                layer = {}
                layer[1] = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
                layer[1] = tf.nn.relu(layer[1])
                for i in range(2, nhl+1):
                    layer[i] = tf.add(tf.matmul(layer[i-1], weights['h%s'%i]), biases['h%s'%i])
                    layer[i] = tf.nn.relu(layer[i])   
                    layer[i] = tf.nn.dropout(layer[i], keep_prob)
                out_layer = tf.matmul(layer[nhl], weights['out'])
                out_final = tf.add(out_layer, biases['out'], name="output")
                return out_final

            w = {}; b = {};
            for i in range(0, len(ttn['network_shape'][keys])-1):
                w[i+1] = np.transpose(ttn['weights'][keys][i]).astype((np.float32))

            weights = {}; biases={}
            for i in range(1, nhl+1):
                weights['h%s'%i] = tf.Variable(w[i])
            weights['out'] = tf.Variable(w[nhl+1])  
            for i in range(1, nhl+1):
                biases['h%s'%i] = tf.Variable(tf.random_normal([int(n_hidden[i])], mean=0.0, 
                                              stddev=np.sqrt(3.0/(n_classes+int(n_hidden[nhl])))))
            biases['out'] = tf.Variable(tf.random_normal([n_classes], mean=0.0, 
                                        stddev=np.sqrt(3.0/(n_classes+int(n_hidden[nhl])))))  

            pred = multilayer_perceptron(x, weights, biases)
            if(n_classes == 1): predictions = tf.reshape(pred, [-1])
            else: predictions = pred

            #for classification, we need to calculate the accuracy
            if regression == False:             
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), "float32"))

            #l2 weight penalty + mse cost fn  
            weight_decay = tf.reduce_sum(weight_reg * tf.stack([tf.nn.l2_loss(weights['h%s'%i]) 
                                         for i in range(1,nhl+1)] ))
            if regression == False:
                cost = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)), 
                              weight_decay, name="cost") 
            else:
                cost = tf.add(tf.reduce_mean(tf.square(y-predictions)), 
                          weight_decay, name="cost") 

            #optimizer = tf.contrib.layers.optimize_loss(loss=cost,
            #    global_step=tf.train.get_global_step(),
            #    learning_rate=lrs[pp], optimizer="Adam")
            optimizer = tf.train.AdamOptimizer(learning_rate=lrs[pp]).minimize(loss=cost,  global_step=tf.train.get_global_step(),name="opt")



            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                for epoch in range(0, 100):
                    avg_cost = 0.0
                    total_batch = int(len(xtrain)/float(batch_size))
                    for i in range(total_batch):
                        indx = np.random.randint(len(xtrain), size=batch_size)
                        batch_x, batch_y = xtrain[indx,:],ytrain[indx]
                        _, c, p = sess.run([optimizer, cost, pred], 
                                  feed_dict={x:batch_x, y:batch_y, keep_prob:dropout_keep_prob})
                        avg_cost += c / total_batch
                    accur[pp][epoch] = avg_cost
            sess.close()    
            errormin[pp] = np.mean(accur[pp, 90:])
        indices = errormin.argsort()[:2]
        minlr = np.min((lrs[indices[0]], lrs[indices[1]])) 
        maxlr = np.max((lrs[indices[0]], lrs[indices[1]])) 
        lrs = np.linspace(minlr, maxlr, 10)

    learnrate = minlr 

    print("Determining number of epochs needed...")
    training_epochs = 3000; pp = 0;
    accur = np.zeros((1, training_epochs))
    optimizer = tf.train.AdamOptimizer(learning_rate=learnrate).minimize(loss=cost,
                global_step=tf.train.get_global_step(),name="opt")
    #optimizer = tf.contrib.layers.optimize_loss(loss=cost,
    #            global_step=tf.train.get_global_step(),
    #            learning_rate=learnrate, optimizer="Adam")

    init = tf.global_variables_initializer()    
    with tf.Session() as sess:
        sess.run(init)
        epoch=0
        converged = False
        while(converged == False):
            avg_cost = 0.
            total_batch = int(len(xtrain)/float(batch_size))
            for i in range(total_batch):
                indx = np.random.randint(len(xtrain), size=batch_size)
                batch_x, batch_y = xtrain[indx,:], ytrain[indx]
                _, c, p = sess.run([optimizer, cost, pred], 
                          feed_dict={x:batch_x, y:batch_y, keep_prob:dropout_keep_prob})
                avg_cost += c / total_batch
            accur[pp][epoch] = avg_cost   
            if((epoch > 200) and (epoch % 10 == 0)): 
                upper = np.mean(accur[pp, epoch-10:epoch])
                middle = np.mean(accur[pp, epoch-20:epoch-10])
                lower = np.mean(accur[pp, epoch-30:epoch-20])
                d1 = 100*abs(upper-middle)/upper
                d2 = 100*abs(middle-lower)/middle
                if((d1 < 5) and (d2 < 5)): 
                    converged = True
                    maxep = epoch
            if(epoch+1 == training_epochs): 
                converged = True 
                print("Warning: Reached max # training epochs:", training_epochs)
                maxep = training_epochs
            epoch += 1    
    print('Optimal learning rate: ', learnrate)
    print('Optimal # epochs: ', maxep)
    print('Optimal batch size: ', batch_size)
    return(batch_size, learnrate, maxep)     
    



def tf_continue_training(regression, xscale, yscale, x1, y1, ntrees, 
                          learnrate, training_epochs, batch_size,
                          dropout_keep_prob, nhl, display_step, 
                          modelname, modelpath, random_state):
    """ Reloads and continues training an existing DJINN model.
        
    Args:
        xscale (object): Input scaler.
        yscale (object): Output scaler.
        x1 (ndarray): Input features.
        y1 (ndarray): Output features.
        ntrees (int): Number of decision trees.
        learn_rate (float): Learning rate for optimizaiton of weights/biases.
        training_epochs (int): Number of epochs to train neural network.
        batch_size (int): Number of samples per batch.
        dropout_keep_prob (float): Probability of keeping neuron "on" in dropout layers.
        nhl (ndarray): Number of hidden layers in neural network.
        display_step (int): Cost is printed every display_steps during training.
        model_name (str): File name for model if 'save_model' is True.
        model_path (str): Location of where the model/files are saved. 
        random_state (int): Sets random seed.

    Returns:
        None. Re-saves trained model.
    """
    nhl = int(nhl)
    model_path=modelpath
    model_name=modelname
    if(y1.size > y1.shape[0]): n_classes = y1.shape[1]
    else: n_classes = 1

    sess = {}
    xtrain = xscale.transform(x1)
    if regression == True:
        if(n_classes == 1): ytrain = yscale.transform(y1.reshape(-1,1)).flatten()
        else: ytrain = yscale.transform(y1)

    #for classification, do one-hot encoding on classes
    if regression == False:
        ytrain=np.equal.outer(y1.flatten(), np.arange(len(np.unique(y1)))).astype("float32")

    nninfo = {}
    nninfo['weights'] = {}; nninfo['biases'] = {}; nninfo['initial_weights'] = {};
    if (random_state): tf.set_random_seed(random_state)
    for pp in range(0, ntrees):
        old_saver = tf.train.import_meta_graph('%s%s_tree%s.ckpt.meta'%(model_path,model_name,pp))
        sess[pp] = tf.Session()
        old_saver.restore(sess[pp], '%s%s_tree%s.ckpt'%(model_path,model_name,pp))
        print("Model %s restored"%pp)

        #Restore tensors from previous session
        x = sess[pp].graph.get_tensor_by_name("input:0")
        y = sess[pp].graph.get_tensor_by_name("target:0")
        keep_prob = sess[pp].graph.get_tensor_by_name("keep_prob:0")
        pred = sess[pp].graph.get_tensor_by_name("prediction:0")
        optimizer = sess[pp].graph.get_operation_by_name("Adam")
        cost = sess[pp].graph.get_tensor_by_name("cost:0")
        #optimizer = tf.train.AdamOptimizer(learning_rate=learnrate).minimize(loss=cost,  global_step=tf.train.get_global_step(),name="Adam")
        weights={}; biases={};
        for i in range(1,nhl+1):
            weights['h%s'%i] = sess[pp].graph.get_tensor_by_name("w%s:0"%i)
            biases['h%s'%i] = sess[pp].graph.get_tensor_by_name("b%s:0"%i)
        weights['out'] = sess[pp].graph.get_tensor_by_name("w%s:0"%(nhl+1))  
        biases['out'] = sess[pp].graph.get_tensor_by_name("b%s:0"%nhl)  
        nninfo['initial_weights']['tree%s'%pp] = sess[pp].run(weights)

        #for classification, we need to calculate the accuracy
        if regression == False:             
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), "float32"))

        #continue training
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(xtrain)/float(batch_size))
            for i in range(total_batch):
                indx = np.random.randint(len(xtrain), size=batch_size)
                batch_x, batch_y = xtrain[indx,:], ytrain[indx]
                _, c, p = sess[pp].run([optimizer, cost, pred], 
                                        feed_dict={x: batch_x, y: batch_y, 
                                        keep_prob:dropout_keep_prob})
                avg_cost += c / total_batch
                # display training progresss
            if epoch % display_step == 0:
                if regression == True:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                #for classification, print cost and accuracy:
                else: 
                    avg_accuracy = sess[pp].run(accuracy, 
                        feed_dict = {x: xtrain, y: ytrain, keep_prob:dropout_keep_prob})
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost),
                           "accuracy=", "{:.3f}".format(avg_accuracy))
        print("Optimization Finished!")

        #save model and nn info 
        save_path = old_saver.save(sess[pp], "%s%s_tree%s.ckpt"%(modelpath, modelname, pp))
        print("Model resaved in file: %s" % save_path)
        nninfo['weights']['tree%s'%pp] = sess[pp].run(weights)
        nninfo['biases']['tree%s'%pp] = sess[pp].run(biases)
        sess[pp].close()
    with open('%sretrained_nn_info_%s.pkl'%(modelpath, modelname), 'wb') as f1:
        cPickle.dump(nninfo, f1)    
    return()
