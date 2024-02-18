"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function

import os

import sys
import neat as neat
from mo_cdpb import CartPole
import visualize
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

def delete_file(file_path):
    # ファイルが存在するかどうかを確認
    if os.path.exists(file_path):
        # ファイルを削除
        os.remove(file_path)
        print(f"ファイル '{file_path}' が削除されました。")
    else:
        print(f"ファイル '{file_path}' は存在しません。")

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        f1 = -1.0; f2 = -1.0; f3 = -1.0
        genome.fitness = -1.0
        
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        action_type={'neural_network':(0,1)}
        cp = CartPole()
        f1, f2, f3 = cp.run_multi_objective(1000, testing=False,
                                            action_type= action_type, controller=net)
        #genome.mo_fitness =[f1]        
        #genome.mo_fitness =[f1, f2]        
        #genome.mo_fitness =[f1, f2, f3] 
        genome.mo_fitness =[f1,f2,f3]
        genome.mo_rank = [-1,-1]
        genome.fitness = f1

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    print("Run Hernan")
    
    # Create the population, which is the top-level object for a NEAT run.
    delete_file('species_data.csv')
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    gen = 2
    winner, pop = p.run(eval_genomes, gen)

    # Display the winning genome.
    print("\n\n*** winner ***")
    print(type(winner))
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    action_type={'neural_network':(0,1)}
    #print(winner_net,action_type)
    # 確認するために、各入力ノードと出力ノードのIDを出力します
    print(f"Input nodes: {config.genome_config.input_keys}")
    print(f"Output nodes: {config.genome_config.output_keys}")
    cp = CartPole()
    f1, f2, f3 = cp.run_multi_objective(1000, testing=False,
                                        action_type= action_type, controller=winner_net)
    print("best fitness ", f1, f2, f3)
    
    
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    visualize.create_evolution_animation('species_data.csv', 'evolution_animation2.mp4')

    
    #visualize.draw_net(config, winner, True)
    
    
    
    #tuika
    
    size = len(p.population)
    pop_keys_list = list(p.population.keys())
    for i in range(0,size):
        k_i = pop_keys_list[i]

        if p.population[k_i].mo_rank[0] == 0:
            winner=p.population[k_i]
            visualize.draw_net(config, winner, True)
            print("\n\n*** winner ***")
            print(type(winner))
            print('\nBest genome:\n{!s}'.format(winner))


            #add
            
            # Show output of the most fit genome against training data.
            print('\nOutput:')
            winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
            action_type={'neural_network':(0,1)}
            #print(winner_net,action_type)
            # 確認するために、各入力ノードと出力ノードのIDを出力します
            print(f"Input nodes: {config.genome_config.input_keys}")
            print(f"Output nodes: {config.genome_config.output_keys}")
            cp = CartPole()
            f1, f2, f3 = cp.run_multi_objective(1000, testing=False,
                                                action_type= action_type, controller=winner_net)
            print("best fitness ", f1, f2, f3)
    
    


    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes, 10)
    
    return p, winner


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'cdpb_config-feedforward.py')
    p, winner = run(config_path)
    print(winner.mo_fitness, winner.fitness)



"""
cp=CartPole()
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'cdpb_config-feedforward.py')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

size = len(p.population)
pop_keys_list = list(p.population.keys())
for i in range(0,size):
    k_i = pop_keys_list[i]

    if p.population[k_i].mo_rank[0] == 0:
        winner=p.population[k_i]
        print("\n\n*** winner ***")
        print(type(winner))
        print('\nBest genome:\n{!s}'.format(winner))


        #add
        
        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        action_type={'neural_network':(0,1)}
        #print(winner_net,action_type)
        # 確認するために、各入力ノードと出力ノードのIDを出力します
        print(f"Input nodes: {config.genome_config.input_keys}")
        print(f"Output nodes: {config.genome_config.output_keys}")
        # cp = CartPole()
        # f1, f2, f3 = cp.run_multi_objective(1000, testing=False,
        #                                     action_type= action_type, controller=winner_net)
        # print("best fitness ", f1, f2, f3)

cart_positions,pole1_angles,pole2_angles,input_values,output_values = cp.run_data(1000, testing=False,action_type= action_type, controller=winner_net)

cp.create_animation(input_values,output_values,cart_positions,pole1_angles,pole2_angles)

"""
