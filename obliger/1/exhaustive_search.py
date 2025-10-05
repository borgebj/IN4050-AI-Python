from utils import city_names, get_path_distance, format_time
from plotter import plotter, show_all_figures
from itertools import permutations
import time


# <----------- MAIN OPTIMIZATION FUNCTION ----------> #
def exhaustive_search(cities, verbose=False):
    """Finds the shortest path among the given permutations of cities
    Uses 'Exhaustive Search'

    cities: all cities we want to permute
    """
    shortest_distance = float("inf")
    shortest_path = None

    num_perms = 0   # permutation counter
    for permutation in permutations(cities):
        num_perms += 1

        total_distance = get_path_distance(permutation, verbose)
        # total_distance = sum(matrix[c_k][c_((k+1)mod n))], n-1, k=0

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
    verbose = True
    LIMIT = 3  # 24 max

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
    print(f"\nShortest path:\n>\t{path_str}\nwith distance:\n>\t{distance:.4f}")
    print(f"Number of permutations checked:\n>\t{format(num_perms, ',d')}")
    print(f"Time taken for {LIMIT} cities:\n>\t{format_time(end - start)}\n")

    # ============ EXTRA =========== #

    plotter(
        LIMIT=10,
        MAX_EXTRAPOLATE=24,
        path=[city_names[i] for i in path],
        function=exhaustive_search,
        extrapolate=True
    )

    show_all_figures()


if __name__ == "__main__":
    main()
