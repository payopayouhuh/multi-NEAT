#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 21:27:21 2022

@author: hernan
"""

# ---------------- #
# Cart pole module #
# ---------------- #
#import sys
from random import randint, random

''' HERNAN
    - dpole.cpp has been modified for python 3 
    - run setup.py as indicated in the setup.py file
    - use python3 instead of python to run the sript
'''

from dpole import integrate # wrapped from C++


class CartPole(object):
    def __init__(self, markov, rand = False):

        # there are two types of double balancing experiment:
        # 1. markovian: velocity information is provided to the network input
        # 2. non-markovian: no velocity is provided
        self.__markov = markov
        self.__state = []
        # according to Stanley (p. 45) the initial state condition is not random
        self.__initial_state_rand = rand
        '''
        state[0] : cart's  position
        state[1] : cart's  speed
        state[2] : pole_1  angle
        state[3] : pole_1  angular velocity
        state[4] : pole_2  angle
        state[5] : pole_2  angular velocity
        '''

    state = property(lambda self: self.__state)

    def run(self, testing=False):
        """ Runs the cart-pole experiment and evaluates the population. """

        self.__initial_state()

        if testing:
            # cart's position, first pole's angle, second pole's angle
            print ("\nInitial conditions:")
            print("%+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f" \
                      %(self.__state[0], self.__state[1], self.__state[2],self.__state[3], self.__state[4], self.__state[5]))
                #pass
        if(self.__markov):
            # markov experiment: full system's information is provided to the network
            
            score = self.__run_markov(1000, testing)

            if testing:
                print ("fitness (score): %d " %score)
            
            return(score,)

        else:
            # non-markovian: no velocity information is provided (only 3 inputs)

            F, score = self.__run_non_markov(1000, testing)
            
            if testing:
                print ("F %1.3f, score %d  " %(F, score))
            
            return(F, score)
                
    def __run_markov(self, max_steps, testing):
        
        steps = 0

        while(steps < 100000):
            
            ''' SET ACTION HERNAN '''
            action = random()

            # advances one time step
            self.__state = integrate(action, self.__state, 1)
            
            # new state
            if testing:
                print("%d %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f" \
                      %(steps+1, self.__state[0], self.__state[1], self.__state[2],
                        self.__state[3], self.__state[4], self.__state[5]))

            if(self.__outside_bounds()):
                # network failed to solve the task
                if testing:
                    print("Failed at step %d " % (steps+1) )
                    break
                else:
                    break
            steps += 1  
            
        return steps
                
    def __run_non_markov(self, max_steps, testing):
        # variables used in Gruau's fitness function
        den = 0.0
        F = 0.0
        last_values = []

        steps = 0
        while(steps < max_steps):
            
            ''' HERNAN action must be properly set'''
            action = random()
            
            self.__state = integrate(action, self.__state, 1)
            
            # new state
            if testing:
                print("%d %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f" \
                      %(steps+1, self.__state[0], self.__state[1], self.__state[2],
                        self.__state[3], self.__state[4], self.__state[5]))

            if(self.__outside_bounds()):
                # network failed to solve the task
                if testing:
                    print ("Failed at step %d " %(steps+1))
                    break
                else:
                    break

            den = abs(self.__state[0]) + abs(self.__state[1]) + \
                  abs(self.__state[2]) + abs(self.__state[3])
            
            last_values.append(den)

            if len(last_values) == 100:
                last_values.pop(0) # we only need to keep the last 100 values

            steps += 1

        # compute Gruau's fitness
        if steps > 100:
            # the denominator is computed only for the last 100 time steps
            jiggle = sum(last_values)
            F = 0.1*steps/1000.0 + 0.9*0.75/(jiggle)
        else:
            F = 0.1*steps/1000.0

        return (F, steps)


    def __initial_state(self):
        """ Sets the initial state of the system. """
        
        # according to Stanley (p. 45) the initial state condition is not random
        
        if self.__initial_state_rand:
            self.__state = [randint(0,4799)/1000.0 - 2.4,  # cart's initial position
                            randint(0,1999)/1000.0 - 1.0,  # cart's initial speed
                            randint(0, 399)/1000.0 - 0.2,  # pole_1 initial angle
                            randint(0, 399)/1000.0 - 0.2,  # pole_1 initial angular velocity
                            randint(0,2999)/1000.0 - 0.4,  # pole_2 initial angle
                            randint(0,2999)/1000.0 - 0.4]  # pole_2 initial angular velocity
        else:
            self.__state = [0.0,
                            0.0,
                            0.07, # set pole_1 to one degree (in radians)
                            0.0,
                            0.0,
                            0.0]

    def __outside_bounds(self):
        """ Check if outside the bounds. """

        failureAngle = 0.628329 #thirty_six_degrees

        return  abs(self.__state[0]) > 2.4 or \
                abs(self.__state[2]) > failureAngle or \
                abs(self.__state[4]) > failureAngle