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


def get_distance(city1, city2):
    """Returns the distance between two cities."""
    i = get_city_index(city1)
    j = get_city_index(city2)

    return matrix[i][j]