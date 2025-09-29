from utils import cities, matrix, permute, get_path_distance
from plotter import plot_plan
import time


def generate_neighbors(path):
    """Generates neighboring paths by swapping two cities in the current path."""

    neighbors = []
    cities = len(path)

    # for every city
    for i in range(cities):
        # swap with every other city after
        for j in range(i + 1, cities):
            neighbor = list(path)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i] # swap
            neighbors.append(tuple(neighbor))

    return neighbors


def find_shortest_path(city_perms):
    """Finds the shortest path among the given permutations of cities.
    Uses 'Hill Climbing'

    city_perms: all permutations of the cities
    """

    #TODO: finn random utvalg
    random_path = city_perms[0]

    neighbors = generate_neighbors(random_path)
    current_shortest = get_path_distance(random_path, verbose)

    print(f"Starting path:\n{random_path} = {current_shortest:.4f}\n")

    for path in neighbors:
        dist = get_path_distance(path, verbose)
        print(f"{path} = {dist:.4f}")


def main():
    global cities, matrix, verbose

    verbose = False2

    # limits no. cities
    LIMIT = 3
    cities = cities[:LIMIT]

    start = time.time()

    # gives all permutations of cities
    city_perms = permute(cities)

    # finds the shortest path among the permutations
    path = find_shortest_path(city_perms)

    end = time.time()
    print(f"\nTime taken: {end - start:.4f} seconds")

    # plot_plan(path)


if __name__ == "__main__":
    main()
