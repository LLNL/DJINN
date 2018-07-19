.. #
.. # Copyright (c) 2018, Lawrence Livermore National Security, LLC.
.. # 
.. # Produced at the Lawrence Livermore National Laboratory
.. #
.. # Written by K. Humbird (humbird1@llnl.gov), L. Peterson (peterson76@llnl.gov).
.. #
.. # LLNL-CODE-754815
.. #
.. # All rights reserved.
.. #
.. # This file is part of DJINN.
.. #
.. # For details, see github.com/LLNL/djinn. 
.. #
.. # For details about use and distribution, please read DJINN/LICENSE .
.. #

Installation Guide
==================
<<<<<<< HEAD
`djinn` requires both scikit-learn (recommend v0.19 or later) 
and tensorflow (v1.0.1 or later).
=======
`djinn` requires both scikit-learn and tensorflow.
>>>>>>> djinn_v1.0_public

Installing with pip
-------------------
You can install djinn via pip.

1. clone the repo from: github.com/LLNL/djinn ::

   $ cd djinn_parent_dir
<<<<<<< HEAD
   $ git clone [link to djinn repo]
=======
   $ git clone < link to repo >
>>>>>>> djinn_v1.0_public

2. pip user install (eg)::

   $ cd djinn
   $ pip install --user .

3. test out loading djinn::

   $ cd
   $ python -m djinn.djinn

4. try out an example::

   $ python djinn_parent_dir/djinn/tests/djinn_example.py 
