from utils import cities, matrix, permute, get_distance
from plotter import plot_plan
import time


def find_shortest_path(city_perms):
    """Finds the shortest path among the given permutations of cities.
    Uses 'Hill Climbing'

    city_perms: all permutations of the cities
    """
    None


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
    print(f"Time taken: {end - start:.4f} seconds")

    plot_plan(path)


if __name__ == "__main__":
    main()
