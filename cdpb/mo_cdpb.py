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
import math
import sys

''' HERNAN
    - dpole.cpp has been modified for python 3 
    - run setup.py as indicated in the setup.py file
    - use python3 instead of python to run the sript
'''

from dpole import integrate # wrapped from C++

# there are two types of double balancing experiment:
# 1. markovian: velocity information is provided to the network input
# 2. non-markovian: no velocity is provided

class CartPole(object):
    def __init__(self, init_state_rand = False):

        self.__FAILURE_ANGLE = math.pi*36/180  #0.62831853 #thirty_six_degrees 
        self.__L = 4.8

        #self.__FAILURE_ANGLE = math.pi
        #self.__L = 6

        
        self.__markov = True
        self.__state = []
        # according to Stanley (p. 45) the initial state condition is not random
        self.__initial_state_rand = init_state_rand

        ''' Initial values and range
        state[0] : cart's  position  0.0 or in [-L/2, L/2] = [-2.4, 2.4] if random initial value 
        state[1] : cart's  speed     0.0 or in [-1,1] if random initial value 
        state[2] : pole_1  angle                1 degree in radians 0.017 or in [-0.2, 0.2] if random initial value
        state[3] : pole_1  angular velocity     0 or in [-0.2, 0.2]  if random initial value 
        state[4] : pole_2  angle                0 or in [-0.4, -0.1] if random initial value 
        state[5] : pole_2  angular velocity     0 or in [-0.4, -0.1] if random initial value 
        '''

    state = property(lambda self: self.__state)

    def run_multi_objective(self, max_steps, testing, action_type={}, controller=None):
        
        self.__initial_state()
        steps = 0
        
        '''
        print("t action cx cv p1a p1v p2a p2v")
        print("%d %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f" \
              %(steps, 0.0, self.__state[0], self.__state[1], self.__state[2],
                        self.__state[3], self.__state[4], self.__state[5]))
        '''




        count_change_direction = 0
        f1 =0.0
        f2 =0.0
        f3 =0.0
        
        positions =[]

        while(steps < max_steps):
            
            action_keys = list(action_type.keys())
            ''' SET ACTION HERNAN '''
            if 'random' in action_keys:
                max_v, min_v = action_type['random']
                action = min_v + random() * (max_v-min_v)
            elif 'neural_network' in action_keys:
                #print("action taken by neural network")
                #print("not implementeed yet")
                xi = [self.__state[0],self.__state[1],self.__state[2], self.__state[4]]
                action = controller.activate(xi)[0]
                #sys.exit()
            else:
                print("Invalid action type")
                sys.exit()
                
            #print("action ", action)

            # advances one time step
            new_state = integrate(action, self.__state, 1)
            
            #did the cart change direction?            
            if steps == 0:
                if new_state[0] - self.__state[0] > 0:
                    direction = 1  # East
                else:
                    direction = -1 # West
            else:
                if ((new_state[0] - self.__state[0])* direction) < 0:
                    if testing:
                        print (" change direction: ", direction, -1*direction, 
                               count_change_direction+1)
                    direction = -1 * direction
                    count_change_direction = count_change_direction + 1
                    
            positions.append(new_state[0])

            self.__state = new_state
            # new state
            if testing:
                print("%d %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f" \
                      %(steps+1, action, self.__state[0], self.__state[1], self.__state[2],
                        self.__state[3], self.__state[4], self.__state[5]))

            if(self.__outside_bounds(testing)):
                # network failed to solve the task
                if testing:
                    print("Failed at step %d " % (steps+1) )
                    break
                else:
                    break
                
            steps += 1  
        
        ''' For minimization
        f1 = (max_steps - steps)/max_steps
        f2 = count_change_direction / steps
        # positions are in the range [-L/2, L/2] [-2.4, 2.4]
        sum_distance_to_center = sum(abs(number) for number in positions)
        f3 = sum_distance_to_center / (steps*L/2) # average distance from center per time unit
        '''
        
        ''' for maximization '''
        # f1 fraction of goal time, f1 in [0.0, 1.0]
        f1 = steps/max_steps
        
        # f2 inverse of number of changes in direction per time unit, 
        # f2 in [1.0, t], t is steps
        # no direction changes t many direction chamges 1.0
        f2 = steps/(count_change_direction + 1) 
        
        # sum distance from the center
        sum_distance_to_center = sum(abs(number) for number in positions) 
        # average distance from center per time unit
        ave = sum_distance_to_center  / (steps*self.__L/2)
        # f3 average distance to the center from the edges per time unit 
        # f3 in [0.0,2.4]
        f3= self.__L/2 - ave  
        
        return f1, f2, f3
    
    def __initial_state(self):
        """ Sets the initial state of the system. """
        
        # according to Stanley (p. 45) the initial state condition is not random
        
        if self.__initial_state_rand:
            self.__state = [randint(0,4799)/1000.0 - self.__L/2,  # cart's initial position
                            randint(0,1999)/1000.0 - 1.0,         # cart's initial speed
                            randint(0, 399)/1000.0 - 0.2,         # pole_1 initial angle
                            randint(0, 399)/1000.0 - 0.2,         # pole_1 initial angular velocity
                            randint(0,2999)/1000.0 - 0.4,         # pole_2 initial angle
                            randint(0,2999)/1000.0 - 0.4]         # pole_2 initial angular velocity
        else:
            self.__state = [0.0,
                            0.0,
                            math.pi*1/180, # set pole_1 to one degree ( 0.017 in radians)
                            0.0,
                            0.0,
                            0.0]

    def __outside_bounds(self, testing=False):
        """ Check if outside the bounds. """

        #failureAngle = 0.628329 #thirty_six_degrees
        
        if testing:
            if abs(self.__state[0]) > self.__L/2:
                print("out of the track", self.__state[0])
            if abs(self.__state[2]) > self.__FAILURE_ANGLE:
                print("pole 1 failure angle %+2.1f"  % (self.__state[2]*180/math.pi))
            if abs(self.__state[4]) > self.__FAILURE_ANGLE :
                print("pole 2 failure angle %+2.1f" % (self.__state[4]*180/math.pi))


        return  abs(self.__state[0]) > self.__L/2 or \
                abs(self.__state[2]) > self.__FAILURE_ANGLE or \
                abs(self.__state[4]) > self.__FAILURE_ANGLE
    
    def generate_data(self, x):
        for force in [1, -1]:
            for k in range(0,101) :
                action = force * k/100
                nx = integrate(action, x, 1)
                
                oc = 0; op1 = 0; op2 = 0
                #if out of bounds  oc, op1 or op2 are set to 1  
                if abs(nx[0]) > self.__L/2 :
                    oc = 1
                if abs(nx[2]) > self.__FAILURE_ANGLE :
                    op1 = 1
                if abs(nx[4]) > self.__FAILURE_ANGLE :
                    op2 = 1
                
                # deviations: car position from center of the track, or 
                # pole angles from the vertical 
                dc = abs(nx[0]- 0.0)/self.__L/2
                dp1 = abs(nx[2])/self.__FAILURE_ANGLE
                dp2 = abs(nx[4])/self.__FAILURE_ANGLE
                if oc + op1 + op2 >= 0:   
                    outf = "%d %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f " + \
                           "%+1.4f %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f " + \
                           "%d %d %d " + \
                           "%+1.4f %+1.4f %+1.4f %+1.4f" 
                    print(outf %(k, x[0],  x[1],  x[2],  x[3],  x[4],  x[5], action, 
                                  nx[0], nx[1], nx[2], nx[3], nx[4], nx[5],
                                  oc, op1, op2,
                                  dc, dp1, dp2, dc+dp1+dp2))
                    
    def run_data(self, max_steps, testing, action_type={}, controller=None):
        
        self.__initial_state()
        steps = 0
        
        '''
        print("t action cx cv p1a p1v p2a p2v")
        print("%d %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f" \
              %(steps, 0.0, self.__state[0], self.__state[1], self.__state[2],
                        self.__state[3], self.__state[4], self.__state[5]))
        '''

        ''' Initial values and range
        state[0] : cart's  position  0.0 or in [-L/2, L/2] = [-2.4, 2.4] if random initial value 
        state[1] : cart's  speed     0.0 or in [-1,1] if random initial value 
        state[2] : pole_1  angle                1 degree in radians 0.017 or in [-0.2, 0.2] if random initial value
        state[3] : pole_1  angular velocity     0 or in [-0.2, 0.2]  if random initial value 
        state[4] : pole_2  angle                0 or in [-0.4, -0.1] if random initial value 
        state[5] : pole_2  angular velocity     0 or in [-0.4, -0.1] if random initial value 
        '''

        cart_positions = []
        pole1_angles = []
        pole2_angles = []
        input_values = []
        output_values = []
        


        count_change_direction = 0
        
        positions =[]

        while(steps < max_steps):
            
            action_keys = list(action_type.keys())
            ''' SET ACTION HERNAN '''
            if 'random' in action_keys:
                max_v, min_v = action_type['random']
                action = min_v + random() * (max_v-min_v)
            elif 'neural_network' in action_keys:
                #print("action taken by neural network")
                #print("not implementeed yet")
                xi = [self.__state[0],self.__state[1],self.__state[2], self.__state[4]]
                action = controller.activate(xi)[0]

                #sys.exit()
            else:
                print("Invalid action type")
                sys.exit()
                
            #print("action ", action)

            # advances one time step
            new_state = integrate(action, self.__state, 1)
            
            #did the cart change direction?            
            if steps == 0:
                if new_state[0] - self.__state[0] > 0:
                    direction = 1  # East
                else:
                    direction = -1 # West
            else:
                if ((new_state[0] - self.__state[0])* direction) < 0:
                    if testing:
                        print (" change direction: ", direction, -1*direction, 
                               count_change_direction+1)
                    direction = -1 * direction
                    count_change_direction = count_change_direction + 1
                    
            positions.append(new_state[0])

            self.__state = new_state
            # new state
            if testing:
                print("%d %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f %+1.4f" \
                      %(steps+1, action, self.__state[0], self.__state[1], self.__state[2],
                        self.__state[3], self.__state[4], self.__state[5]))

            if(self.__outside_bounds(testing)):
                # network failed to solve the task
                if testing:
                    print("Failed at step %d " % (steps+1) )
                    break
                else:
                    break
            
            cart_positions.append(self.__state[0])
            pole1_angles.append(self.__state[2])
            pole2_angles.append(self.__state[4])
            input_values.append(xi)
            output_values.append(action)

                
            steps += 1  
        

        
        return cart_positions,pole1_angles,pole2_angles,input_values,output_values
    
    def create_animation(self, input_values, output_values, cart_positions, pole1_angles, pole2_angles):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots()  # サブプロットを1つだけ作成
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 2)

        # カートとポールの描画用のサブプロット
        cart, = ax.plot([], [], 'ks-', lw=2)
        pole1, = ax.plot([], [], 'ro-', lw=4)
        pole2, = ax.plot([], [], 'bo-', lw=4)

        # 入力値と出力値の表示用のテキストオブジェクト
        info_text = ax.text(-2.8, 1.7, '', fontsize=8)  # グラフの上部に配置

        # カートとポールの描画ロジック
        def draw_cart_pole(cart_position, pole1_angle, pole2_angle):
            cart_x = cart_position
            cart_y = 0
            pole1_x = cart_x + 1.0 * np.sin(pole1_angle)
            pole1_y = 1.0 * np.cos(pole1_angle)
            pole2_x = cart_x + 0.3 * np.sin(pole2_angle)
            pole2_y = 0.3 * np.cos(pole2_angle)
            cart.set_data([cart_x], [cart_y])
            pole1.set_data([cart_x, pole1_x], [cart_y, pole1_y])
            pole2.set_data([cart_x, pole2_x], [cart_y, pole2_y])

        # アニメーションフレームごとの更新ロジック
        def animate(frame):
            inputs = input_values[frame]
            output = output_values[frame]
            info = f"t={frame}\nInputs: {inputs}\nOutput: {output}"
            info_text.set_text(info)

            # カートとポールの描画
            cart_position = cart_positions[frame]
            pole1_angle = pole1_angles[frame]
            pole2_angle = pole2_angles[frame]
            draw_cart_pole(cart_position, pole1_angle, pole2_angle)

        ani = FuncAnimation(fig, animate, frames=len(cart_positions), interval=50)
        ani.save('cart_pole_animation.gif', writer='pillow')

        plt.show()
