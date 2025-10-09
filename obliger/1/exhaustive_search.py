from utils import city_names, path_distance, format_time
from plotter import plot_plan
from statistics import plot_extrapolation
from itertools import permutations
import time


# <----------- MAIN OPTIMIZATION FUNCTION ----------> #
def exhaustive_search(cities, verbose=False):
    """Finds the shortest path among the given permutations of cities
    Uses 'Exhaustive Search'

    cities: list of city indices
    """
    shortest_distance = float("inf")
    shortest_path = None

    num_perms = 0  # permutation counter
    for permutation in permutations(cities):
        num_perms += 1

        total_distance = path_distance(permutation, verbose)

        # compare shortest
        if total_distance < shortest_distance:
            shortest_distance = total_distance
            shortest_path = permutation

        # display in terminal
        if verbose:
            print(f"\nTotal distance: {total_distance:.4f} km")

    return list(shortest_path), shortest_distance, num_perms


def main():
    # main flags
    verbose = False
    LIMIT = 10  # 24 max

    # limits no. cities
    cities = list(range(LIMIT))  # represents cities as indexes

    # all permutations, then finds shortest among all
    start = time.time()
    path, distance, num_perms = exhaustive_search(cities, verbose=verbose)
    end = time.time()

    # prints info on main run
    # indices turned back to string
    path_names = [city_names[i] for i in path]
    path_str = ' -> '.join(path_names) + f" -> {path_names[0]}"
    print("\n=== Results ===\n")
    print(f"Shortest path:\n    {path_str}\n")
    print(f"Distance:\n    {distance:.4f}\n")
    print(f"Number of permutations checked:\n    {format(num_perms, ',d')}\n")
    print(f"Time taken for {LIMIT} cities:\n    {format_time(end - start)}\n")

    # ============ EXTRA =========== #
    plot_plan(path_names)

    plot_extrapolation(
        limit=10,
        max_extrapolate=24,
        function=exhaustive_search,
        extrapolate=True
    )


if __name__ == "__main__":
    main()
