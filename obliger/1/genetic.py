from utils import city_names, path_distance, format_time
from statistics import run_statistics
import matplotlib.pyplot as plt
import random
import time


# <----------- MAIN OPTIMIZATION FUNCTIONS ----------> #

class Individual:
    """Class representing an individual in the population.
    Structure of individual inspired by geeksforgeeks implementation: https://www.geeksforgeeks.org/dsa/genetic-algorithms/
    """

    def __init__(self, path):
        self.path = path
        self.distance = path_distance(path)
        self.fitness = 1 / self.distance
        # this way, higher fitness is better (shorter distance)

    def create_random(cities, rng):
        path = cities.copy()
        rng.shuffle(path)
        return Individual(path)

    def mutate(self, rng=None):
        rng = rng or random

        mutation_type = rng.randint(0, 2)  # 0, 1, 2 types
        start, end = generate_segment(len(self.path), rng)

        # swap mutation     (swap 2 random indices)
        if mutation_type == 0:
            i, j = rng.sample(range(len(self.path)), 2)
            self.path[i], self.path[j] = self.path[j], self.path[i]

        # inversion     (reverse random segment)
        elif mutation_type == 1:
            self.path[start:end] = reversed(self.path[start:end])

        # shuffle       (separate, shuffle, insert)
        elif mutation_type == 2:
            segment = self.path[start:end]
            rng.shuffle(segment)
            self.path[start:end] = segment

        # recalculate fitness
        self.distance = path_distance(self.path)
        self.fitness = 1 / self.distance if self.distance > 0 else float('inf')

    def mate(self, other, rng=None):
        rng = rng or random
        childpath = order_crossover(self.path, other.path, rng)
        return Individual(childpath)

    # for printing purposes
    def __repr__(self):
        return f"{self.path}"

    def __str__(self):
        return f"Path: {self.path}\tDistance: {self.distance:.2f}\tFitness: {(self.fitness * 1000):.3f}"


def generate_segment(size, rng=None):
    """Generate index segment within 'size'"""
    rng = rng or random
    if size < 2: return 0, size

    # max half length segment
    segment_len = max(2, size // 2)

    # anywhere within bounds
    start = rng.randint(0, size - segment_len)
    end = start + segment_len

    return start, end


def generate_population(cities, pop_size, rng):
    """
    Generates 'pop size' number of permutations for initial population
    using individual class and random shuffling
    """
    population = []

    # ensures no duplicates by storing seen individuals
    seen = set()
    while len(population) < pop_size:
        individual = Individual.create_random(cities, rng)
        individual_tuple = tuple(individual.path)

        if individual_tuple not in seen:
            seen.add(individual_tuple)
            population.append(individual)

    return population


def order_crossover(parent1, parent2, rng):
    """Performs ordered crossover between two parents"""
    offspring = [None] * len(parent1)
    size = len(parent1)

    # 1. choose crossover segment, roughly half
    start, end = generate_segment(size, rng)

    # 2. copies segment over
    offspring[start:end] = parent1[start:end]

    # 3. fill blanks from parent 2, skip duplicates, continues after segment
    current_idx = end % size
    for city in parent2:
        if city not in offspring:
            # find next blank and wrap around
            offspring[current_idx] = city
            current_idx = (current_idx + 1) % size

    return offspring


def tournament_selection(population, rng, k=3):
    """Picks k random individuals, returns best based on fitness"""
    k = min(len(population), k)

    # chooses k random
    sample = rng.sample(population, k)

    # finds best, returns it
    best = sample[0]
    for ind in sample:
        if ind.fitness > best.fitness:
            best = ind

    return best


def genetic_algorithm(cities, pop_size=None, seed=None, verbose=False):
    """
    Main GA function
    implementing generational model with:
    - Elitism (top ~10%)
    - Ordered Crossover (OX)
    - Mix of mutation operators (swap, inversion, shuffle)
    - Fixed max generation (based on N)
    """
    if seed:
        rng = random.Random(seed)  # for reproducibility
    else:
        rng = random.Random()

    # Hyperparameters:
    n = len(cities)
    if not pop_size:
        pop_size = max(50, n * 5)  # population size (e.g. 120 for 24 cities)
    if n < 10:
        pop_size = max(10, n * 2)  # for smaller samples

    tournament_k = n // 6  # k random looked at for parents
    crossover_prob = 0.9  # 90% crossover chance
    mutation_prob = 0.15  # 15% mutation chance
    elite_count = 0.1  # 0.1 as in 10% of best carries on
    max_generations = max(50, n * 15)  # max no. generations (e.g. 360 for 24 cities)

    # Step 1 - generate initial population
    population = generate_population(cities, pop_size, rng)
    sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    overall_best = sorted_population[0]

    # [ (generation, best fit), ... ]
    fitness_stats = []

    if verbose:
        print(f"\nInitial population:")
        print(*population, sep="\n")
        print(f"\nBest:\n{overall_best}\n")

    # termination condition : generational loop
    for generation in range(1, max_generations + 1):

        # Step 2 - elitism - keeps top ~10% of best solutions
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        elites = sorted_pop[:max(1, int(elite_count * pop_size))]
        new_population = elites.copy()

        if verbose:
            print(f"\n\n\n{'=' * 30}[ Gen {generation} start ]{'=' * 30}")
            print(f"Elites carried over:")
            for i, elite in enumerate(elites, start=1):
                print(f"{i}:\t    {elite}")
            print()

        # Step 3 - fill population with offspring
        while len(new_population) < pop_size:

            # Tournament selection   (ensures not duplicate)
            parent1 = tournament_selection(population, rng, k=tournament_k)
            p2_pool = [ind for ind in population if ind.path != parent1.path]
            if not p2_pool: p2_pool = population
            parent2 = tournament_selection(p2_pool, rng, k=tournament_k)

            # Crossover - chance: 90%
            if rng.random() < crossover_prob:
                offspring1 = parent1.mate(parent2, rng)
                offspring2 = parent2.mate(parent1, rng)
            else:
                # 10% chance of same individuals "surviving"
                offspring1 = Individual(parent1.path.copy())
                offspring2 = Individual(parent2.path.copy())

            # Mutation - 15% chance
            if rng.random() < mutation_prob:
                offspring1.mutate(rng)
            if rng.random() < mutation_prob:
                offspring2.mutate(rng)

            if verbose:
                print(f"Parent1:    {parent1}")
                print(f"Parent2:    {parent2}")
                print(f"Offspring1: {offspring1}")
                print(f"Offspring2: {offspring2}\n")

            # New population
            new_population.append(offspring1)
            if len(new_population) < pop_size:
                new_population.append(offspring2)

        # Step 3 - replace old generation
        population = new_population

        # Step 4 - update best solution
        gen_best = max(population, key=lambda ind: ind.fitness)
        fitness_stats.append((generation, round(gen_best.distance, 4)))

        if gen_best.fitness > overall_best.fitness:
            overall_best = gen_best

        if verbose:
            print(f"Gen best:   {gen_best}")

    return overall_best.path, overall_best.distance, fitness_stats

def plot_average_fitness(curves, labels):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    runs = len(curves[0])

    # one figure
    plt.figure(figsize=(8, 5))

    # curve_set = [run1, run2, run3]
    for i, (curve_set, label) in enumerate(zip(curves, labels)):
        run_values = {}

        for run in curve_set:
            for (gen, distance) in run:
                run_values[gen] = run_values.get(gen, 0) + distance

        # calculate average distance for each generation
        run_averages = {gen: dist / runs for gen, dist in run_values.items()}

        x = list(run_averages.keys())
        y = list(run_averages.values())

        # add this sets data to plot
        plt.plot(
            x, y,
            label=f"Pop. {label}",
            color=colors[i % len(colors)],
            linewidth=1.5,
            alpha=0.8
        )

    plt.xlabel("Generation")
    plt.ylabel("Average Distance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    # main flags
    verbose = False
    limit = 20
    seed = random.randint(0, 2 ** 16 - 1)

    # limits no. cities
    cities = list(range(limit))  # represents cities as indexes

    start = time.time()
    path, distance, best_fitness_gens = genetic_algorithm(cities, seed=seed, verbose=verbose)
    end = time.time()

    # prints info on main run
    path_names = [city_names[i] for i in path]
    path_str = ' -> '.join(path_names) + f" -> {path_names[0]}"
    print(f"\n\n\nShortest path:\n>\t{path_str}\n\nwith distance:\n>\t{distance:.4f}\n")
    print(f"Number of generations:\n>\t{len(best_fitness_gens)}\n")
    print(f"Time taken for {limit} cities:\n>\t{format_time(end - start)}\n")
    print(f"Seed used:\n>\t{seed}\n")

    # statistics (worst, mean) + plot           (lambda prevents it from running first)
    # population growth: ( n*5, n*10, n*15 )
    cities = list(range(24))
    sizes = [40, 120, 360]
    curves1 = run_statistics(lambda: genetic_algorithm(cities, pop_size=sizes[0]), f"Genetic 24/{sizes[0]}")
    curves2 = run_statistics(lambda: genetic_algorithm(cities, pop_size=sizes[1]), f"Genetic 24/{sizes[1]}")
    curves3 = run_statistics(lambda: genetic_algorithm(cities, pop_size=sizes[2]), f"Genetic 24/{sizes[2]}")

    plot_average_fitness([curves1, curves2, curves3], sizes)


if __name__ == "__main__":
    main()
