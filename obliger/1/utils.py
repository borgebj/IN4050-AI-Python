import csv
from itertools import permutations

"""Reads the CSV file and returns the cities and distance matrix."""
with open("european_cities.csv", "r") as f:
    data = list(csv.reader(f, delimiter=';'))
    cities = data[0]

    matrix = data[1:]
    # removing row 1 (labels), now works as an adjacency matrix
    # matrix[i][j] is the distance from city i to city j


def permute(cities):
    """Returns all permutations of the given list of cities."""
    return list(permutations(cities))


def get_city_index(city):
    """Returns the index of the given city in the data."""
    return cities.index(city)


def get_city_distance(city1, city2):
    """Returns the distance between two cities."""
    i = get_city_index(city1)
    j = get_city_index(city2)

    return matrix[i][j]


def get_path_distance(path, verbose):
    """Calculates the total distance of the given path."""
    total_distance = 0
    num_cities = len(path)

    # all cities in a permutation
    for i, city in enumerate(path):
        next_city = path[(i + 1) % num_cities]  # wraps around to first
        dist = get_city_distance(city, next_city)
        total_distance += float(dist)

        # display in terminal
        if verbose:
            print(f"\t{city:8}\t-> {dist} ->\t{next_city},")

    return total_distance
