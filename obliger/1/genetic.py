from typing import List

from utils import city_names, get_path_distance, format_time
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

    # step 1, generate population
    population = generate_population(cities)

    print(f"All cities:\n{city_names}")
    print(f"Initial population:")
    [print(f"{i}\t{lst}") for i, lst in enumerate(population)]

    # placeholder
    best_path = population[0]
    best_distance = get_path_distance(best_path)
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
