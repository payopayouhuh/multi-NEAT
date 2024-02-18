#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 21:33:50 2022

@author: hernan
"""

# Compile the C++ Python extension for the
# cart-pole experiment:

# python 2
# python setup.py build_ext -i

# python 2
# python3 setup.py build_ext -i
from distutils.core import setup, Extension
setup(
      name='Cart-pole experiment',
      ext_modules=[
               Extension('dpole', ['dpole.cpp'])]               
)
