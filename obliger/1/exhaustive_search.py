from utils import cities, matrix, permute, get_path_distance, format_time
from plotter import plotter
import numpy as np
import math
import time




# <----------- MAIN OPTIMIZATION FUNCTION ----------> #
def exhaustive_search(cities, verbose=False):
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

    # main flags
    verbose = False
    LIMIT = 8

    # limits no. cities
    cities = cities[:LIMIT]

    # all permutations, then finds shortest among all
    start = time.time()
    path, distance = exhaustive_search(cities, verbose=verbose)
    end = time.time()

    # prints info on main run
    path_str = ' -> '.join(path) + f" -> {path[0]}"
    print(f"\nShortest path:\n>\t{path_str}\nwith distance:\n>\t{distance:.4f}")
    num_perms = math.factorial(len(cities))
    print(f"Number of permutations checked:\n>\t{format(num_perms, ',d')}")
    print(f"Time taken for {LIMIT} cities:\n>\t{format_time(end - start)}\n")



    # ============ EXTRA =========== #

    plotter(
            LIMIT=8, 
            MAX_EXTRAPOLATE=24, 
            path=path, 
            function=exhaustive_search,
            extrapolate=True
    )


if __name__ == "__main__":
    main()
