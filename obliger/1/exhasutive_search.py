import csv
from itertools import permutations
import time
from plotter import plot_plan


"""Reads the CSV file and returns the cities and distance matrix."""
with open("european_cities.csv", "r") as f:
    data = list(csv.reader(f, delimiter=';'))
    cities = data[0]

    matrix = data[1:]
    # removing row 1 (labels), now works as an adjacency matrix
    # matrix[i][j] is the distance from city i to city j

    verbose = None


def permute(cities):
    """Returns all permutations of the given list of cities."""
    return list(permutations(cities))


def get_city_index(city):
    """Returns the index of the given city in the data."""
    return cities.index(city)


def get_distance(city1, city2):
    """Returns the distance between two cities."""
    i = get_city_index(city1)
    j = get_city_index(city2)

    return matrix[i][j]


def find_shortest_path(city_perms):
    """Finds the shortest path among the given permutations of cities.

    city_perms: all permutations of the cities
    data:
    """
    shortest_distance = float("inf")
    shortest_path = None

    # all permutations
    for permutation in city_perms:
        total_distance = 0

        # all cities in a permutation
        for i, city in enumerate(permutation):
            next_city = permutation[(i + 1) % len(permutation)]
            dist = get_distance(city, next_city)
            total_distance += float(dist)

            if verbose:
                print(f"\t{city:8}\t-> {dist} ->\t{next_city},")

        # compare shortest
        if total_distance < shortest_distance:
            shortest_distance = total_distance
            shortest_path = permutation

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
    print(f"Time taken: {end - start:.4f} seconds")

    plot_plan(path)


if __name__ == "__main__":
    main()
