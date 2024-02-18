#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 09:45:32 2022

@author: hernan
"""

''' x1 x2 AND OR XOR
    0  0   0  0  0 
    0  1   0  1  1
    1  0   0  1  1
    1  1   1  1  0
'''


''' TEST DOUBLE CART POLE BALANCING '''

from cdpb_KAWABATA import CartPole

population = None
simulation = CartPole(markov = False)
#simulation = CartPole(markov = False)
simulation.print_status = True
for i in range(10):
    print("--- ", i, " ---")
    simulation.run(True)
