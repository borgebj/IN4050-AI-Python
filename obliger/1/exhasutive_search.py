import time
from plotter import plot_plan
from utils import cities, matrix, permute, get_path_distance


def find_shortest_path(city_perms):
    """Finds the shortest path among the given permutations of cities.
    Uses 'Exhaustive Search'

    city_perms: all permutations of the cities
    """
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
            print(f"Total distance: {total_distance:.4f} km")
            print(f"\n")

    path = ' -> '.join(shortest_path) + f" -> {shortest_path[0]}"
    print(f"\nShortest path:\n>\t{path}\nwith distance\n>\t{shortest_distance:.4f}\n")
    print(f"Number of permutations checked: {format(len(city_perms), ',d')}")
    return shortest_path


def main():
    global cities, matrix, verbose

    verbose = False

    # limits no. cities
    LIMIT = 10
    cities = cities[:LIMIT]

    start = time.time()

    # gives all permutations of cities
    city_perms = permute(cities)

    # finds the shortest path among the permutations
    path = find_shortest_path(city_perms)

    end = time.time()
    print(f"\nTime taken: {end - start:.4f} seconds")

    plot_plan(path)


if __name__ == "__main__":
    main()
