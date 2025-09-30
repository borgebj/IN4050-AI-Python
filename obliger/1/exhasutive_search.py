from utils import cities, matrix, permute, get_path_distance, format_time, measure_runtime, extrapolate_runtime
from plotter import plot_plan, plot_times, show_all_figures
import numpy as np
import math
import time


# <----------- EXTRA ----------> #



# <----------- MAIN OPTIMIZATION FUNCTION ----------> #
def exhasutive_search(cities, verbose=False):
    """Finds the shortest path among the given permutations of cities
    Uses 'Exhaustive Search'

    cities: all cities we want to permute
    """

    city_perms = permute(cities)

    shortest_distance = float("inf")
    shortest_path = None

    # all permutations
    for permutation in city_perms:

        total_distance = get_path_distance(permutation, verbose)

        # compare shortest
        if total_distance < shortest_distance:
            shortest_distance = total_distance
            shortest_path = permutation

        # display in terminal
        if verbose:
            print(f"Total distance: {total_distance:.4f} km\n")

    if verbose:
        path = ' -> '.join(shortest_path) + f" -> {shortest_path[0]}"
        print(f"\nShortest path:\n>\t{path}\nwith distance:\n>\t{shortest_distance:.4f}")
        print(f"Number of permutations checked:\n>\t{format(len(city_perms), ',d')}\n")

    return list(shortest_path), shortest_distance



def main():
    global cities, matrix

    # flags
    verbose = False
    LIMIT = 8

    # limits no. cities
    cities = cities[:LIMIT]

    # all permutations, then finds shortest among all
    start = time.time()
    path, distance = exhasutive_search(cities, verbose=verbose)
    end = time.time()

    # prints info on main run
    path_str = ' -> '.join(path) + f" -> {path[0]}"
    print(f"\nShortest path:\n>\t{path_str}\nwith distance:\n>\t{distance:.4f}")
    num_perms = math.factorial(len(cities))
    print(f"Number of permutations checked:\n>\t{format(num_perms, ',d')}")
    print(f"Time taken for {LIMIT} cities:\n>\t{format_time(end - start)}\n")



    # ============ EXTRA =========== #

    # plotting
    LIMIT = 8
    MAX_EXTRAPOLATE = 24
    
    # 1. Measure
    times_measured = measure_runtime(exhasutive_search, LIMIT, step=1)

    # 2. Extrapolate
    times_extrapolated, predict = extrapolate_runtime(times_measured, MAX_EXTRAPOLATE, "exhaustive")

    print("\nPredicted times for values:\n")
    for n in [5, 10, 15, 20, 24]:
        t_sec = float(predict(n))
        print(f"{n:2d} cities: {format_time(t_sec)}")

    # 3. plot
    plot_times(times_measured, times_extrapolated, exhasutive_search)

    plot_plan(path)

    show_all_figures()


if __name__ == "__main__":
    main()
