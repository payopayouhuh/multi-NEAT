"""
Handles creation of genomes, either from scratch or by sexual or
asexual reproduction from parents.
"""
from __future__ import division

import math
import random
from itertools import count
from math import ceil

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean
from neat.six_util import iteritems, itervalues
### NSGA2
from neat.population import print_nds
from neat.population import print_fronts
from neat.population import fronts_ranking
from neat.population import crowding_distance
from neat.population import non_dominated_sorting
from neat.population import get_non_dominated_solutions
from neat.population import dominates

### add 2023/01/25
def dominates(find1, find2):
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

def get_non_dominated_solutions(old_members):
    size = len(old_members)
    dom_count = [0] * size

    # 非支配カウント
    for i in range(size):
        _, find_i = old_members[i]
        for j in range(size):
            if i != j:
                _, find_j = old_members[j]
                if dominates(find_i.mo_fitness, find_j.mo_fitness):
                    dom_count[j] += 1
    
    # 非支配ソリューションを抽出
    ndpop = []
    for i in range(size):
        if dom_count[i] == 0:
            ndpop.append(old_members[i])
    
    return ndpop


def non_dominated_sorting(old_members):
    fronts = []
    while old_members:
        ndpop = get_non_dominated_solutions(old_members)
        fronts.append(ndpop)
        old_members = [member for member in old_members if member not in ndpop]
    return fronts



def crowding_distance(front, nobj, i_index=1):
    nind = len(front)
    INFINITY = 1e+15
    EPS = 1e-10

    # Initialize cd = 0 to for all individuals in the front
    for _, genome in front:
        genome.mo_rank[i_index] = 0.0

    for i in range(nobj):
        # sort front by ith objective 
        front.sort(reverse=True, key=lambda x: x[1].mo_fitness[i])

        # get max and min fitness in the front
        maxf = front[0][1].mo_fitness[i]
        minf = front[-1][1].mo_fitness[i]

        # add a very large distance to extreme solutions
        front[0][1].mo_rank[i_index] += INFINITY
        front[-1][1].mo_rank[i_index] += INFINITY
        # add distance in objective i from solution j-1 to j+1
        for j in range(1, nind-1):
            d = front[j-1][1].mo_fitness[i] - front[j+1][1].mo_fitness[i]
            front[j][1].mo_rank[i_index] += abs(d)/abs(EPS + maxf - minf)


import sys

def fronts_ranking(fronts, nobj=2, i_index=0):
    for i in range(len(fronts)):
        front = fronts[i]
        # 各フロント内でゲノムを処理
        for j in range(len(front)):
            genome = front[j][1]
            genome.mo_rank[i_index] = i
        crowding_distance(front, nobj, i_index=1)


def mo_ranking(old_members):
    fronts = non_dominated_sorting(old_members)
    fronts_ranking(fronts, nobj=2)

    # 適応度の更新
    for front in fronts:
        for _, genome in front:
            genome.fitness = genome.mo_rank[0]
            genome.fitness += 1.0 / (1 + genome.mo_rank[1])
            genome.fitness *= -1.0

###

# TODO: Provide some sort of optional cross-species performance criteria, which
# are then used to control stagnation and possibly the mutation rate
# configuration. This scheme should be adaptive so that species do not evolve
# to become "cautious" and only make very slow progress.

class DefaultReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2)])

    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        #種ごとの適正な子孫数を計算するフィットネスに比例する
        #目的の個体群サイズから、適合度 (0 ～ 1 のスケールで調整) に従って、種ごとに目的のメンバー数を配分
        af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
            c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1

            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

        return spawn_amounts

    def reproduce(self, config, species, pop_size, generation):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """
        # TODO: I don't like this modification of the species and stagnation objects,
        # because it requires internal knowledge of the objects.

        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.
        #停滞した種をフィルタリングし、停滞していない種のメンバーの集合を集め、その平均調整済みフィットネスを計算する。
        #平均調整済みフィットネス方式(区間[0, 1]に正規化)は、共有フィットネス方式を妨げることなく負のフィットネス値を使用することを可能にする。
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
        #stagnation.update(species, generation):
        #種の適応度の履歴情報を更新し、 max_stagnation世代で改善されていないものをチェックし、-
      #削除された場合に設定された種の数が種のエリート主義を下回る結果にならない限り (その場合、最高の適応度の種は免れる) を返す。
      #除去対象としてマークされた停滞種のリスト。
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in itervalues(stag_s.members))
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {} # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in itervalues(afs.members)])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses) # type: float
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size,self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)

        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            #もしエリート主義が有効なら、それぞれの種は常に少なくともそのエリートを保持することができる

            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            #この種には少なくとも次の世代のためのメンバーがいるので、それを保持します。
            old_members = list(iteritems(s.members))
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.

            # 種毎にフロント分け実装 2025/01/25
            mo_ranking(old_members)


            old_members.sort(reverse=True, key=lambda x: x[1].fitness)


            # NEAT
            # elite_count = self.reproduction_config.elitism

            # NSGA2
            elite_count = ceil(spawn / 2) 

            # Using ceil to round up if spawn is odd

            # Transfer elites to new generation.
            #エリートを新世代に移譲する
            if self.reproduction_config.elitism > 0:

                #NSGA2 elite_count
                #for i, m in old_members[:self.reproduction_config.elitism]:
                for i, m in old_members[:elite_count]:
                    new_population[i] = m
                    spawn -= 1
                    #spawn += 1 #NSGA2_2


            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *
                                         len(old_members)))
            

            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)


            #repro_cutoff→survival_threshold = 0.5 より　NSGA2 choiceは適正
            old_members = old_members[:repro_cutoff]



            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population
