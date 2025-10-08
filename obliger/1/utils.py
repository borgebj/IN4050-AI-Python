import csv
import time
from typing import Literal
import numpy as np
import math

"""Reads the CSV file and returns the cities and distance matrix."""
with open("european_cities.csv", "r") as f:
    data = list(csv.reader(f, delimiter=';'))
    city_names = data[0]

    matrix = [[float(x) for x in row] for row in data[1:]]
    # removing row 1 (labels), now works as an adjacency matrix
    # matrix[i][j] is the distance from city i to city j


def city_distance(city1, city2):
    """Returns the distance between two cities."""
    return matrix[city1][city2]


def path_distance(path, verbose=False):
    """Calculates the total distance of the given path."""
    total_distance = 0
    num_cities = len(path)

    # all cities in a permutation
    for i, city in enumerate(path):
        next_city = path[(i + 1) % num_cities]
        distance = city_distance(city, next_city)
        total_distance += distance

        # display in terminal
        if verbose:
            print(f"\t{city:8}\t-> {distance} ->\t{next_city},")

    return total_distance


# ====================== TIMING ======================= #

def format_time(seconds):
    """Used to display times from extrapolated values"""
    if seconds < 60:
        return f"{seconds:.4f} s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.4f} min"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.4f} h"
    days = hours / 24
    if days < 365:
        return f"{days:.4f} d"
    years = days / 365
    if years > 1e6:
        return f"{years:.2e} y"
    return f"{years:.4f} y"


def measure_runtime(function, n):
    """Gets runtime of given function for values 1 to n cities (n!)"""
    times = []

    for i in range(1, n + 1):
        cities = list(range(i))

        start = time.time()
        res = function(cities)
        end = time.time()

        time_taken = end - start
        times.append((i, time_taken))

    return times


def extrapolate_exhaustive(times, n):
    """Extrapolates time taken for n cities based on measured times
        Does so by calculating how much time used per permutation,
        averaging all for an estimated constant factor

        Returns extrapolated times and a function to predict any n
    """
    x = np.array([t[0] for t in times])  # x vals (no. cities)
    y = np.array([t[1] for t in times])  # y vals (time in s)

    # k = time / n!
    # how much time per permutation
    k_values = [y / math.factorial(x) for x, y in zip(x, y)]

    # average all per-permutation times for extrapolation
    k = np.mean(k_values)

    print(f"\n === Extrapolation Model ===")
    print(f"Time(x) = {k} * x!")

    def predict(x):
        return k * math.factorial(x)

    extrapolate_times = [
        (i, predict(i)) for i in range(max(x)+1, n+1)
    ]

    return extrapolate_times, predict

