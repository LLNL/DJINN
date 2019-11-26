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
# For details about use and distribution, please read DJINN/LICENSE.
###############################################################################

from setuptools import setup
from setuptools import find_packages
from os import listdir

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='djinnml',
      version='1.0',
      description='Deep Jointly-Informed Neural Networks',
      long_description=readme(),
      classifiers=[
                   'Programming Language :: Python :: 3.7',
                   ],
      keywords='regression neural networks tensorflow',
      url='https://github.com/LLNL/djinn',
      author='Kelli Humbird',
      author_email='humbird1@llnl.gov',
      license='LLNL',
      packages=find_packages(),
      install_requires=[
                        'tensorflow>=2.0.0',
                        'scipy',
                        'scikit-learn>=0.21',
                        ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      #entry_points={ 'console_scripts': find_scripts(),
      #},
      #scripts=bin_scripts(),
      include_package_data=True,
      zip_safe=False)
