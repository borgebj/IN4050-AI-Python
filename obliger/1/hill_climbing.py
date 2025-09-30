from utils import cities, matrix, get_path_distance, format_time, measure_runtime, extrapolate_runtime
from plotter import plot_plan, plot_times, show_all_figures
import numpy as np
import random
import math
import time


# <----------- EXTRA ----------> #






# <----------- MAIN OPTIMIZATION FUNCTIONs ----------> #
def generate_start(cities):
    """Takes a list of cities and shuffles it"""
    random.shuffle(cities)
    return cities
    

def generate_neighbors(path):
    """Generates neighboring paths by swapping two cities in the current path."""

    neighbors = []
    cities = len(path)

    # for every city
    for i in range(cities):
        # swap with every other city after
        for j in range(i + 1, cities):
            neighbor = path.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i] # swap
            neighbors.append(neighbor)

    return neighbors


def hill_climb(cities, verbose=False):
    """Finds the shortest path among the given permutations of cities.
    Uses 'Hill Climbing'

    cities: all cities we want to look through
    """

    # chooses an arbitrary (random) start, as well as its distance
    start = generate_start(cities)
    current_shortest = get_path_distance(start)

    step = 0     # step counter
    while True:
        neighbors = generate_neighbors(start)

        best_neighbor = min(neighbors, key=get_path_distance)
        best_distance = get_path_distance(best_neighbor)

        if verbose:
            print(f"\n[{step}] Start: {start} ({current_shortest:.2f})")
            print(f"[{step}] Best:  {best_neighbor} ({best_distance:.2f})\n")


        if best_distance < current_shortest:
            start = best_neighbor
            current_shortest = best_distance
            step += 1
        else:
            break

    return start, current_shortest, step


def main():
    global cities, matrix

    # flags
    verbose = False
    LIMIT = 24

    # limits no. cities
    cities = cities[:LIMIT]

    # finds the shortest path with regards to neighboring paths
    start = time.time()
    path, distance, step = hill_climb(cities, verbose=verbose)
    end = time.time()

    # prints info on main run
    path_str = ' -> '.join(path) + f" -> {path[0]}"
    print(f"\nShortest path:\n>\t{path_str}\nwith distance:\n>\t{distance:.4f}")
    print(f"Total neighbors visited:\n>\t{step+1}")
    print(f"Time taken for {LIMIT} cities:\n>\t{format_time(end - start)}\n")



    # ============ EXTRA =========== #

    # plotting
    LIMIT = 12
    MAX_EXTRAPOLATE = 24

    # 1. Measure
    times_measured = measure_runtime(hill_climb, LIMIT, step=1)

    # 2. Extrapolate
    times_extrapolated, predict = extrapolate_runtime(times_measured, MAX_EXTRAPOLATE, "hill")

    print("\nPredicted times for values:\n")
    for n in [5, 10, 15, 20, 24, 28, 32, 36, 40, 44, 48]:
        t_sec = float(predict(n))
        print(f"{n:2d} cities: {format_time(t_sec)}")

    plot_times(times_measured, times_extrapolated, hill_climb)

    plot_plan(path)

    show_all_figures()



if __name__ == "__main__":
    main()
