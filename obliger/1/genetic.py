from typing import List

from utils import city_names, path_distance, format_time
from plotter import plotter
import random
import numpy as np
import time


# <----------- MAIN OPTIMIZATION FUNCTIONS ----------> #

def generate_population(cities: List[int]):
    """Generates 'pop size' number of permutations for initial population"""
    pop_size = len(cities) * 5
    population = []
    seen_permutations = set()

    # generate random permutations, add to population
    while len(population) < pop_size:

        perm = tuple(np.random.permutation(cities))

        if perm not in seen_permutations:
            population.append(list(perm))
            seen_permutations.add(perm)

    return population


def genetic_algorithm(cities, verbose=False):
    """
    Main GA function
    Does selection, crossover, mutation and returns best result
    """
    # params:
    # pop_size = 5 * n   or   10 * n
    # generations = 500   or   scale = 30 * n^(0.8)
    # tournament_k = 3-5
    # crossover_prob = 0.9
    # mutation_prob 0.15

    # representation:
    # - individual = permutation of city inidces
    # - path_distance(path) = distance between all cities
    # - path_fitness(path) = 1 / path_distance(path)

    # step 1 - generate population
    population = generate_population(cities)

    # step 2 - evaluate fitness
    # - compute + cache fitness[i] and distance[i]

    # step 3 - tournament selection
    # - pick k random
    # - top 2 become parent (parent1, parent2)

    # step 4 - crossover (PMX)(0.9?)
    # - start idx, end idx, within len(path)
    # -- segment[start:end]
    # - 90% of crossover
    # - use parents from tep 2
    # - create new offspring (child1, child)

    # step 5 - mutation (0.15?)
    # - 15% of mutation
    # - random combination of methods?
    # - shuffle, inversion, swap
    # - compute distance + fitness, then cache

    # step 6 - elitism
    # - 2 choices:
    # - Explicit elitism (keep e.g. 10%, fill rest with repeated offspring)
    # - (mu + lambda)  (repeated offspring fill, combine, choose pop_size best)

    print(f"All cities:\n{city_names}")
    print(f"Initial population:")
    [print(f"{i}\t{lst}") for i, lst in enumerate(population)]

    # placeholder
    best_path = population[0]
    best_distance = path_distance(best_path)
    epochs = 0

    return best_path, best_distance, epochs


def main():
    # main flags
    verbose = False
    LIMIT = 4

    # limits no. cities
    cities = list(range(LIMIT))  # represents cities as indexes

    start = time.time()
    path, distance, epoch = genetic_algorithm(cities, verbose=verbose)
    end = time.time()

    # prints info on main run
    # path_str = ' -> '.join(path) + f" -> {path[0]}"
    # print(f"\nShortest path:\n>\t{path_str}\nwith distance:\n>\t{distance:.4f}")
    # print(f"Number of epochs:\n>\t{epoch}")
    # print(f"Time taken for {LIMIT} cities:\n>\t{format_time(end - start)}\n")


if __name__ == "__main__":
    main()


"""
# Setup
pop = [random_permutation(n) for _ in range(pop_size)]
fitness = [fitness_of(ind) for ind in pop]
best = argmax(fitness)

for gen in 1..generations:
    new_pop = []
    # elitism: carry top elites
    elites = top_k(pop, fitness, elitism_count)
    new_pop.extend(elites)

    while len(new_pop) < pop_size:
        # tournament selection (sample k, pick best)
        p1 = tournament(pop, fitness, k=tournament_k)
        p2 = tournament(pop, fitness, k=tournament_k)

        # crossover
        if random() < crossover_prob:
            c1, c2 = pmx(p1, p2)
        else:
            c1, c2 = p1.copy(), p2.copy()

        # mutation (swap)
        if random() < mutation_prob: swap_mutation(c1)
        if random() < mutation_prob: swap_mutation(c2)

        # evaluate children
        fitness_c1 = fitness_of(c1)
        fitness_c2 = fitness_of(c2)

        new_pop.append(c1)
        if len(new_pop) < pop_size:
            new_pop.append(c2)

    # update population and best
    pop = new_pop
    fitness = [fitness_of(ind) for ind in pop]  # or maintain incremental updates
    current_best = argmax(fitness)
    if fitness[current_best] > fitness[best]:
        best = current_best
        # optionally store solution
    # optional early stopping: no improvement for patience gens -> break
"""