
'''
    Python-only script to evaluate a trained DJINN model. DJINN must be trained with "save_files" set to True; this script requires the nn_info_%modelname.pkl file that is created after training the tensorflow-based model.
    
'''

import numpy as np

def relu(x):
    ''' ReLU function.
        Args:
        x (ndarray): Data.
        Returns:
        ReLU(x)
        '''
    return(np.max((np.zeros(x.shape),x), axis=0))


def evaluate_djinn(x, modelname, modelpath):
    ''' Evaluates the trained djinn model.
        Args:
        x (ndarray): Input vector/array (n_samples, n_inputs).
        modelname (string): Name given to djinn model.
        Returns:
        (ndarray): Reconstructed representation of z.
        '''
    #need to scale inputs and unscale outputs...
    
    import cPickle
    with open("%snn_info_%s.pkl"%(modelpath,modelname), "rb") as f:
        d=cPickle.load(f)
    
    input_minmax = d['input_minmax']
    output_minmax = d['output_minmax']

    x = (x-input_minmax[0])/(input_minmax[1]-input_minmax[0])

    weights=d['final_weights']['tree_0']
    biases=d['final_biases']['tree_0']
    num_layers=len(weights.keys())
    out = x
    for j in range(1,num_layers):
        out = np.dot(out, weights['h%s'%j])+biases['h%s'%j]
        out = relu(out)
    out = np.dot(out, weights['out'])+biases['out']

    out = output_minmax[0]+out*(output_minmax[1]-output_minmax[0])

    return(out)
