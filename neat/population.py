"""Implements the core evolution algorithm."""
from __future__ import print_function
import sys
from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
#%matplotlib inline
from matplotlib import pyplot as pyp
import os
sys.path.append(r"C:\Users\kawabata\AppData\Roaming\Python\Python37\site-packages\neat")
from hv import HyperVolume
from species import gdmean_list, gdstdev_list

all_gdmean_list = []
all_gdstdev_list = []


#import ea_base as ea
#add dominates

# NAGA2 以外で検証　シミュレーション検討
# 構造の複雑度検証
# アルゴリズムちゃんと記述


def dominates(find1, find2):
    """Determines whether fitness vector find1 dominates 
    fitness vector find2
    If find1 dominates find2 returns True
    Otherwise returns False
    
    Parameters
    ----------
    find1: touple or list
        Vector of fitness values of individual 1
    find2: touple or list
        Vector of fitness values of individual 2
            
    Returns
    -------
    Boolean
        True if find1 dominates find2. False otherwise
         
    """
    dom = False
    better = 0
    better_or_equal = 0
    nobj = len(find1)
    
    for i in range(0,nobj):
        if (find1[i] >= find2[i]):
            better_or_equal += 1
            if find1[i] > find2[i]:
                better += 1
    
    if (better_or_equal == nobj) and better >= 1:
        dom = True
    return dom

def get_non_dominated_solutions(pop):
    """Extract non-dominated solutions from pop and return them
    Dominated solutions remain in pop
    
    Parameters
    ----------
    pop: Population dictionary {id: genome,...}
        Population fron which the set of non-dominated solutions will be 
        extracted
            
    Returns
    -------
    Population
        The set of non-dominated solutions, dictionary
   
    """
    size = len(pop)
    dom_count = [0] * size
    pop_keys_list = list(pop.keys())

    """
    Count how many times a solution has been dominated
    """
    for i in range(0,size):
        k_i = pop_keys_list[i]
        find_i = pop[k_i].mo_fitness
        for j in range(0,size):
            if i != j:
                k_j = pop_keys_list[j]
                find_j = pop[k_j].mo_fitness
                if dominates(find_i, find_j):
                    dom_count[j] += 1
    
    """
    A solutions i with dom_count[i] == 0 is non-dominated
    """
 
    ndpop = {}
    for i in range(0,size):
        if dom_count[i] == 0:
            # remove non-dominated solution from pop
            k = pop_keys_list[i]
            v = pop.pop(k,None)
            # add non-dominated solution to ndpop
            ndpop[k]=v
    return ndpop

def non_dominated_sorting(pop):
    """Extract non-dominated fronts from pop and include them in a list
    The first front in the lits is the top front
    pop is empty after all fronts have been extracted
    Returns the list of non-diminated fronts
  
    Parameters
    ----------
    pop: Population dictionary {id: genome,...}
        Population fron which sets of non-dominated solutions will be 
        extracted
            
    Returns
    -------
    List
        The sets of non-dominated solutions
    """

    fronts = []
    while (len(pop) > 0):
        fi  = get_non_dominated_solutions(pop)
        fronts.append(fi)
    return fronts


def crowding_distance (front, nobj,i_index=1):
    """Computes crowding distance of a set of non-dominated solutions
 
    Parameters
    ----------
    front: Population
        Front of solutions to compute crowding distance
    nobj: integer
        The number of objectives
    i_index:
        The index of the rank corresponding to crowding distance. 
        Default is 1
        
    Returns
    -------
    None    
    """
    
    nind = len(front)
    INFINITY = 1e+15
    EPS = 1e-10
    
    front_items = list(front.items())

    # Initialize cd = 0 to for all individuals in the front
    for i in range(0, nind):
        front_items[i][1].mo_rank[i_index] = 0.0

    for i in range (0, nobj):
        # sort front by ith objective 
        front_items.sort(reverse=True, key=lambda x: x[1].mo_fitness[i])

        # get max and min fitness in the front
        maxf = front_items[0][1].mo_fitness[i]
        minf = front_items[nind-1][1].mo_fitness[i]

        # add a very large distance to extreme solutions
        front_items[0][1].mo_rank[i_index] += INFINITY
        front_items[nind-1][1].mo_rank[i_index] += INFINITY
        # add distance in objective i from solution j-1 to j+1
        for j in range(1, nind-1):
            d = front_items[j-1][1].mo_fitness[i] - front_items[j+1][1].mo_fitness[i]
            front_items[j][1].mo_rank[i_index] += abs(d)/abs(EPS + maxf - minf)

import sys

def fronts_ranking(fronts, nobj=2, i_index=0):
    """Ranks all solutions of a front with the specified drank 
    A solutions can have two ranks: rank[0] and rank[0]
    By default, i_index=0  so drank is assigned to rank[0]

    Parameters
    ----------
    front: Population dictionary {id : genome, ..., }
        Fronts of solutions to be ranked
        i_index:
    The index of the rank corresponding to front number 
        Default is 0
  
            
    Returns
    -------
    None
    
    """
    # all solutions in the same front are assigned the same rank
    # the rank = front number
    for i in range(len(fronts)):
        front_items = list(fronts[i].items())
        for j in range(len(front_items)):
            genome = front_items[j][1]
            genome.mo_rank[i_index] = i
        crowding_distance (fronts[i], nobj,i_index=1)

def print_fronts(fronts):
    i_obj=0
    for i in range(len(fronts)):
        print("front ", i, " size ", len(fronts[i]))
        fi = list(fronts[i].items())
        fi.sort(reverse=True, key=lambda x: x[1].mo_fitness[i_obj])
        for j in range(len(fi)):
            genome_id = fi[j][0]
            genome = fi[j][1]
            print(genome_id, genome.mo_fitness[0], genome.mo_fitness[1], 
                  genome.mo_rank[0], genome.mo_rank[1])


def print_nds(pop, t):
    print("--- Non dominated solutions ---")
    #print("type pop", type(pop))
    ipop_all = list(pop.items())
    ipop=[]
    for x in ipop_all:
        if x[1].mo_fitness is not None:
            ipop.append(x)
    ipop.sort(reverse=True, key=lambda x: x[1].mo_fitness[0])

    directory = "ndpop"
    file_path = os.path.join(directory, "ndpop" + str(t) + ".txt")
    with open(file_path, "w") as fp:
        # fp= open("ndpop"+str(t)+".txt", "w")
    #    for gid, genome in ipop:
        for i in range(len(ipop)):
            #print("type ipop ", type(ipop[i]))
            #print("ipop[i] ", ipop[i])
            #print(ipop[i][1])
            gid   = ipop[i][0]
            genome= ipop[i][1]
            #print("print_nds ", gid)
            if genome.mo_rank is not None:
                if genome.mo_rank[0] < 1:
                    #print(i)
                    fitness = ""
                    for k in range(len(genome.mo_fitness)):
                        fitness = fitness + " " + str(genome.mo_fitness[k])
                    #print(gid, genome.fitness, genome.mo_fitness[0], genome.mo_fitness[1], 
                    #      genome.mo_rank[0], genome.mo_rank[1])
                    print(gid, genome.fitness, fitness, 
                        genome.mo_rank[0], genome.mo_rank[1])
                    wstr =  str(gid) + " "+str(genome.fitness) + " " + fitness +" "
                    wstr += str(genome.mo_rank[0]) + " " +str(genome.mo_rank[1]) +"\n"
                    fp.write(wstr)
    fp.close()
    

class CompleteExtinctionException(Exception):
    pass


class Population(object):
    #object継承
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        #コンストラクタ 「インスタンス化されたときに最初に呼ばれる特別なメソッド」
        #設定の読み込み
        self.reporters = ReporterSet()
        self.config = config
        
        #種が進歩しているかどうかを追跡し、 構成可能な世代数の間、そうでないものを削除する
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        
        
        #クラスreproduction.DefaultReproduction(構成、レポーター、停滞
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)

      
        # ゲノム適合度のセットから終了基準を計算するために使用される関数 fitness_criterion
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            # ゼロから母集団を作成し、種に分割する。初期集団生成
            
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            #create_new(ゲノムタイプ、ゲノム構成、ゲノム数)
            #指定された構成を使用して、指定されたタイプの新しいゲノムを作成
            #また、祖先情報を (空のタプルとして) 初期化
            #戻り値：	作成されたゲノムの辞書 (一意のゲノム識別子をキーとする)
            #戻り値のタイプ:dict(整数、インスタンス)
            
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            #ゲノム距離に基づいて集団を種に分割(設定読み込み)
            #クラスreproduction.Species(鍵、世代)
            #種を表し、メンバー、フィットネス、停滞時間などの種に関するデータを含む

            self.generation = 0
            gdmean,gdstdev = self.species.speciate(config, self.population, self.generation)
            all_gdmean_list.append(gdmean)
            all_gdstdev_list.append(gdstdev)


            #speciate(構成、人口、世代)
            #遺伝的類似性 (ゲノム距離) によってゲノムを種に分類
            
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)
        
    def mo_ranking(self):

        fronts = non_dominated_sorting(self.population)
        fronts_ranking(fronts)

        for i in range(len(fronts)):
            self.population.update(fronts[i])

        
        genomes = list(self.population.values())
        print("mo_ranking")
        #print(genomes)
        for i in range (len(genomes)):

            # NSGA2
            genomes[i].fitness  = genomes[i].mo_rank[0] 
            genomes[i].fitness += 1.0/(1+genomes[i].mo_rank[1])
            genomes[i].fitness *= -1.0
            
            #F1*F2"*F3...
            #genomes[i].fitness = genomes[i].mo_fitness[0]*genomes[i].mo_fitness[1]*genomes[i].mo_fitness[2]
    """
    def eval_genomes_fitness(self): #no use
        size = len(self.population)
        
        pop_keys_list = list(self.population.keys())

        for i in range(0,size):
            k_i = pop_keys_list[i]
            rank = self.population[k_i].mo_rank[0]
            mo = self.population[k_i].mo_rank[1]
            self.population[k_i].fitness = -(rank + 1/(1+mo))

    """
        

    def run(self, fitness_function, n=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """
        
        Hyper_Volume=[]
        referencePoint=[0,0,0]
        hv = HyperVolume(referencePoint)
        generation=[]
        all_fitness1 = [] # 各世代の目的関数1の値のリスト
        all_fitness2 = [] # 各世代の目的関数2の値のリスト
        all_fitness3 = [] # 各世代の目的関数3の値のリスト
        all_num_connections = []
        all_num_nodes = []
        all_num_connectuins_nodes = []

        def calc_list_nega(lst):
            return [-i for i in lst]
        
        
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1



            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)), self.config)
            #iteritems（）各列のラベルと内容を返す list()関数：文字列やタプルなどの別オブジェクトをリストに変換
            
            "ここに追加"
            self.mo_ranking()
            
            
            print("--- generation ", k, " non-dominated population ---")
            print_nds(self.population, k)
            
            

            #HV compute
    
            front=[]
            fitness1=[]
            fitness2=[]
            fitness3=[]


            if k % 1 == 0:
                size = len(self.population)
                pop_keys_list = list(self.population.keys())
                front = []  # 各世代ごとにfrontをリセットします。
                
                for i in range(size):
                    k_i = pop_keys_list[i]

                    if self.population[k_i].mo_rank[0] == 0:
                        list_f = []
                        f1 = self.population[k_i].mo_fitness[0]
                        f2 = self.population[k_i].mo_fitness[1]
                        f3 = self.population[k_i].mo_fitness[2]


                        fitness1.append(f1)
                        fitness2.append(f2)
                        fitness3.append(f3)

                        list_f.extend([f1, f2, f3])
                        list_f = calc_list_nega(list_f)  
                        front.append(list_f)

                volume = hv.compute(front)  # この計算は各世代ごとに一度だけ実行
                Hyper_Volume.append(volume)
                generation.append(k)
                all_fitness1.append(fitness1)
                all_fitness2.append(fitness2)
                all_fitness3.append(fitness3)

            # 残りの処理...
            #Hyper
            
            if k == n:
                pyp.title("HyperVolume", {"fontsize": 25})
                pyp.xlabel("generation", {"fontsize": 15})
                pyp.ylabel("HyperVolume", {"fontsize": 15})
                pyp.grid(True)
                pyp.plot(generation, Hyper_Volume)
                pyp.legend()
                plot_file_path = 'hypervolume_plot.png'
                pyp.savefig(plot_file_path)
                print(f"Plot saved to {plot_file_path}")
                pyp.show()


                min_fitness1=0
                max_fitness1=0.3
                min_fitness2=0
                max_fitness2=300
                min_fitness3=2
                max_fitness3=3
                


                # 3Dプロット
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                # 軸の範囲を設定（必要に応じて調整してください）
                ax.set_xlim([min_fitness1, max_fitness1])         
                ax.set_ylim([min_fitness2, max_fitness2])
                ax.set_zlim([min_fitness3, max_fitness3]) 
                ax.scatter(fitness1, fitness2, fitness3, c="red")
                ax.set_xlabel("fitness1")
                ax.set_ylabel("fitness2")
                ax.set_zlabel("fitness3")
                ax.set_title("Final Generation")
                plt.savefig("Final_3D_Scatter_Plot.png")



                # (f1, f2)プロット
                plt.figure()
                plt.xlim([min_fitness1, max_fitness1])         
                plt.ylim([min_fitness2, max_fitness2])
                plt.scatter(fitness1, fitness2, c="red")
                plt.xlabel("fitness1")
                plt.ylabel("fitness2")
                plt.title("f1 and f2 in Final Generation")
                plt.savefig('final_f1_f2_plot.png')

                # (f1, f3)プロット
                plt.figure()
                plt.xlim([min_fitness1, max_fitness1]) 
                plt.ylim([min_fitness3, max_fitness3]) 
                plt.scatter(fitness1, fitness3, c="red")
                plt.xlabel("fitness1")
                plt.ylabel("fitness3")
                plt.title("f1 and f3 in Final Generation")
                plt.savefig('final_f1_f3_plot.png')

                # (f2, f3)プロット
                plt.figure()   
                plt.xlim([min_fitness2, max_fitness2])
                plt.ylim([min_fitness3, max_fitness3]) 
                plt.scatter(fitness2, fitness3, c="red")
                plt.xlabel("fitness2")
                plt.ylabel("fitness3")
                plt.title("f2 and f3 in Final Generation")
                plt.savefig('final_f2_f3_plot.png')



                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                
                min_fitness1=0
                max_fitness1=0.3
                min_fitness2=0
                max_fitness2=300
                min_fitness3=2
                max_fitness3=3
                
                # 軸の範囲を設定（必要に応じて調整してください）
                ax.set_xlim([min_fitness1, max_fitness1])         
                ax.set_ylim([min_fitness2, max_fitness2])
                ax.set_zlim([min_fitness3, max_fitness3]) 

                def update_plot(frame_number, all_fitness1, all_fitness2, all_fitness3, plot, gen_text, generation):
                    # このフレームのfitness1, fitness2, fitness3を取得
                    fitness1 = all_fitness1[frame_number]
                    fitness2 = all_fitness2[frame_number]
                    fitness3 = all_fitness3[frame_number]

                    plot[0].remove()
                    plot[0] = ax.scatter(fitness1, fitness2, fitness3, c="red")
                    gen_text.set_text('Generation: {}'.format(generation[frame_number]))

                plot = [ax.scatter([], [], [])]
                # 世代数を表示するテキストオブジェクトを追加
                gen_text = ax.text2D(0.02, 0.05, '', transform=ax.transAxes)

                ani = animation.FuncAnimation(fig, update_plot, frames=len(generation), fargs=(all_fitness1, all_fitness2, all_fitness3, plot,gen_text,generation))

                plt.xlabel("fitness1")
                plt.ylabel("fitness2")
                ax.set_zlabel("fitness3")
                plt.title("Objectives Over Generations")

                # アニメーションをGIFとして保存
                # ani.save('evolution_animation.gif', writer='imagemagick')
                # アニメーションをMP4として保存
                ani.save('evolution_animation.mp4', writer='ffmpeg')




            
            if k % 1 == 0:
                size = len(self.population)
                pop_keys_list = list(self.population.keys())
                num_connections = []  # 各世代ごとにfrontをリセットします。
                num_nodes = []
                num_connectuins_nodes = []
                
                for i in range(size):
                    k_i = pop_keys_list[i]

                    # if self.population[k_i].mo_rank[0] == 0:
                    list_f = []
                    num_con = len(self.population[k_i].connections)
                    num_nod = len(self.population[k_i].nodes) - 1
                    num_con_nod = num_con + num_nod


                    num_connections.append(num_con)
                    num_nodes.append(num_nod)
                    num_connectuins_nodes.append(num_con_nod)

                all_num_connections.append(mean(num_connections))
                all_num_nodes.append(mean(num_nodes))
                all_num_connectuins_nodes.append(mean(num_connectuins_nodes))


            if k == n:
                plt.clf()
                plt.title("Average Number of Connections and Nodes per Generation", fontsize=25)
                plt.xlabel("Generation", fontsize=15)
                plt.ylabel("Average Number", fontsize=15)
                plt.grid(True)

                # Plot each metric on the same graph for comparison
                plt.plot(generation, all_num_connections, label='Average Connections')
                plt.plot(generation, all_num_nodes, label='Average Nodes')
                plt.plot(generation, all_num_connectuins_nodes, label='Average Connections + Nodes')

                # Set the limit for the y-axis if required
                plt.ylim(0, max(max(all_num_connections), max(all_num_nodes), max(all_num_connectuins_nodes)) + 10)

                # Add a legend to explain which line corresponds to which metric
                plt.legend()

                # Save the plot to a file
                plot_file_path = 'combined_metrics_plot.png'
                plt.savefig(plot_file_path)
                print(f"Plot saved to {plot_file_path}")

                # Display the plot
                plt.show()


            if k == n:
                plt.clf()
                plt.title("Genetic Diversity over Generations", fontsize=25)
                plt.xlabel("Generation", fontsize=15)
                plt.ylabel("Diversity Metrics", fontsize=15)
                plt.grid(True)

                # 平均遺伝的距離と標準偏差をプロット
                plt.plot(generation, all_gdmean_list, label='Genetic Distance Mean')
                plt.plot(generation, all_gdstdev_list, label='Genetic Distance Std Dev')

                # y軸のリミットを設定する（必要に応じて）
                plt.ylim(0, max(max(all_gdmean_list), max(all_gdstdev_list)) + 10)

                # 凡例を追加
                plt.legend()

                # グラフをファイルに保存
                plot_file_path = 'genetic_diversity_plot.png'
                plt.savefig(plot_file_path)
                print(f"Plot saved to {plot_file_path}")

                # グラフを表示
                plt.show()

            print("gdmean_list gdstdev_list generation",gdmean_list,gdstdev_list,generation)


         



            #self.eval_genomes_fitness()
            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
            #itervalues()ディクショナリの値に対する反復子を返す *ここの
                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                "fitness閾値を変更する必要あり"
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)
            

            #NSGA2_2
            """
            if k>=2:
                print(k)
                fitness_function(list(iteritems(self.population)), self.config)
                self.mo_ranking()
                # ソートしてカット
                sorted_population = sorted(self.population.items(), key=lambda x: x[1].fitness, reverse=True)
                cut_index = len(sorted_population) // 2  # 上位50%の個体のインデックス（整数値）
                top_population = dict(sorted_population[:cut_index])  # 上位50%の個体だけを残す
                
                self.population = top_population
                
            """


            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            # self.species.speciate(self.config, self.population, self.generation)
            gdmean,gdstdev = self.species.speciate(self.config, self.population, self.generation)
            all_gdmean_list.append(gdmean)
            all_gdstdev_list.append(gdstdev)

            # self.reporters.end_generation(self.config, self.population, self.species)
            
            
            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)
        
        
        pyp.legend()
        pyp.show()

        return self.best_genome, self.population
