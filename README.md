
Deep Jointly-Informed Neural Networks
======================================
**DJINN: Deep jointly-informed neural networks**

DJINN is an easy-to-use algorithm for training deep neural networks on supervised regression tasks. 
For additional information, refer to the paper "Deep neural network initialization with decision trees", cited below. 



Getting Started
-----------
DJINN requires TensorFlow (v1.0.1 or later) and  
scikit-learn (v0.18 or later is recommended).
DJINN also uses numpy, matplotlib, and cPickle.
Sphinx is required to view the html documentation.

Note that the sklearn version used when training a DJINN model must be
the same version used when reloading/evaluating the saved model. 

To use DJINN, clone the repo and install: 

    $ git clone https://github.com/LLNL/DJINN.git
    $ cd DJINN
    $ pip install -r requirements.txt
    $ pip install .


Try it out! 
Examples for training DJINN models are included in the [tests](./tests) folder. 

 -python [djinn_example.py](./tests/djinn_example.py) (single output)

 -python [djinn_multiout_example.py](./tests/djinn_multiout_example.py) (multiple outputs)


For Mac users with Anaconda installs, it might be necessary to manually install matplotlib via pip:

    $ pip install matplotlib


If matplotlib will not import, try running "pythonw", for example: 

    $ pythonw djinn_example.py
    
**Python3 and Tensorflow2**
There is a branch (tf2-py3) that offers a python3 friendly version of djinn that runs on tensorflow2. You can make the tensorflow1 version of djinn python3 friendly by modifying the "djinn.py" script by changing: 
    $ from djinn_fns 
    to 
    $ from .djinn_fns 


### Documentation
To view the DJINN documentation: 

```
cd docs
make html
```
Open docs/_build/html/index.html in a browser


Source Repo
-----------

DJINN is available at https://github.com/LLNL/DJINN


Citing DJINN
-----------
If you use DJINN in your research, please cite the following paper:

K. D. Humbird, J. L. Peterson and R. G. Mcclarren, "Deep Neural Network Initialization With Decision Trees," in IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 5, pp. 1286-1295, May 2019.
doi: 10.1109/TNNLS.2018.2869694,
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8478232&isnumber=8695188




Release 
-----------
Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 
Produced at the Lawrence Livermore National Laboratory

Written by K. Humbird (humbird1@llnl.gov), L. Peterson (peterson76@llnl.gov).

LLNL-CODE-754815   OCEC-18-117

All rights reserved.

Unlimited Open Source- BSD Distribution. 

For release details and restrictions, please read the RELEASE, LICENSE, and NOTICE files, linked below:
- [RELEASE](./RELEASE)
- [LICENSE](./LICENSE)
- [NOTICE](./NOTICE)

