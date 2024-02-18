# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:38:58 2022

@author: kawabata
"""

import warnings

import graphviz
#from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np
import neat as neat
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


def plot_species(statistics, view=False, filename='speciation.svg'):

    # 色分け検討

    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    print("species_sizes",species_sizes)
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        #node_names = {}
        node_names = {
    -1: 'Input cart position',
    -2: 'Input cart    speed',
    -3: 'Input pole_1  angle',
    -4: 'Input pole_2  angle',
    0: 'Output F'
    }
        ''' Initial values and range
        state[0] : cart's  position  0.0 or in [-L/2, L/2] = [-2.4, 2.4] if random initial value 
        state[1] : cart's  speed     0.0 or in [-1,1] if random initial value 
        state[2] : pole_1  angle                1 degree in radians 0.017 or in [-0.2, 0.2] if random initial value
        state[3] : pole_1  angular velocity     0 or in [-0.2, 0.2]  if random initial value 
        state[4] : pole_2  angle                0 or in [-0.4, -0.1] if random initial value 
        state[5] : pole_2  angular velocity     0 or in [-0.4, -0.1] if random initial value 
        '''

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)
    
    #f1 = genome.mo_fitness[0]
    #f2 = genome.mo_fitness[1]
    #f3 = genome.mo_fitness[2]

    #print("-----------------------------------")
    #print(f1,f2,f3)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        #name = node_names.get(k, str(k))
        name = node_names[k]
        input_attrs = {'style': 'filled', 'shape': 'circle', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        #name = node_names.get(k, str(k))
        name = node_names[k]
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue
        #add
        #node_label = "Node {0}\nBias {1}".format(n, genome.nodes[n].bias)
        node_label = "Node {0}\nBias {1}\nActivation: {2}".format(n, genome.nodes[n].bias, genome.nodes[n].activation)

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n),label=node_label, _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))


            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            #add
            edge_label = "{:.2f}".format(cg.weight)
            dot.edge(a, b, label=edge_label,_attributes={'style': style, 'color': color, 'penwidth': width})

    



    dot.render(filename, view=view)

    return dot
def create_evolution_animation(csv_file, output_file='evolution_animation.mp4'):
    # CSVファイルからデータを読み込む
    df = pd.read_csv(csv_file, header=None)
    df.columns = ['generation', 'species_id', 'individual_id', 'f1', 'f2', 'f3']
    # generation + 1, sid, gid, f1, f2, f3,mo_rank,fitness,connections,nodes

    # 世代ごとにデータを整理
    generations = df['generation'].max() + 1
    all_fitness1 = [df[df['generation'] == i]['f1'].values for i in range(generations)]
    all_fitness2 = [df[df['generation'] == i]['f2'].values for i in range(generations)]
    all_fitness3 = [df[df['generation'] == i]['f3'].values for i in range(generations)]
    species_ids = [df[df['generation'] == i]['species_id'].values for i in range(generations)]

    # アニメーションの作成と設定
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 軸の範囲を設定
    min_fitness1 = 0
    max_fitness1 = 0.3
    min_fitness2 = 0
    max_fitness2 = 300
    min_fitness3 = 2
    max_fitness3 = 3
    ax.set_xlim([min_fitness1, max_fitness1])
    ax.set_ylim([min_fitness2, max_fitness2])
    ax.set_zlim([min_fitness3, max_fitness3])

    gen_text = ax.text2D(0.02, 0.05, '', transform=ax.transAxes)

    # アニメーションのための更新関数
    def update_plot(frame, all_fitness1, all_fitness2, all_fitness3, species_ids, gen_text):
        ax.clear()
        ax.set_xlim([min_fitness1, max_fitness1])
        ax.set_ylim([min_fitness2, max_fitness2])
        ax.set_zlim([min_fitness3, max_fitness3])

        gen_text = ax.text2D(0.02, 0.05, '', transform=ax.transAxes)

        for sid in set(species_ids[frame]):
            indices = [i for i, x in enumerate(species_ids[frame]) if x == sid]
            ax.scatter(np.take(all_fitness1[frame], indices),
                       np.take(all_fitness2[frame], indices),
                       np.take(all_fitness3[frame], indices))
        gen_text.set_text('Generation: {}'.format(frame))
        plt.xlabel("fitness1")
        plt.ylabel("fitness2")
        ax.set_zlabel("fitness3")

    
    ani = animation.FuncAnimation(fig, update_plot, frames=generations, fargs=(all_fitness1, all_fitness2, all_fitness3, species_ids, gen_text))

    # タイトルの設定
    plt.title("Objectives Over Generations")

    # アニメーションをMP4ファイルとして保存
    ani.save(output_file, writer='ffmpeg')

    # プロットを閉じる
    plt.close()
