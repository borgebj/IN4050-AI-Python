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


def get_city_distance(city1, city2):
    """Returns the distance between two cities."""
    return matrix[city1][city2]


def get_path_distance(path, verbose=False):
    """Calculates the total distance of the given path."""
    total_distance = 0
    num_cities = len(path)

    # all cities in a permutation
    for i, city in enumerate(path):
        next_city = path[(i + 1) % num_cities]
        distance = get_city_distance(city, next_city)
        total_distance += distance

        # display in terminal
        if verbose:
            print(f"\t{city:8}\t-> {distance} ->\t{next_city},")

    return total_distance


# ====================== TIMING ======================= #

def format_time(seconds):
    """Used to display times from extraploated values"""
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


def measure_runtime(function, n, step=1):
    """Gets runtime of given function for values 1 to n cities (n!)"""
    times = []

    for i in range(1, n + 1, step):
        if function.__name__ == "exhaustive_search":
            cities = list(range(i))
        else:
            cities = list(range(len(city_names)))

        start = time.time()
        res = function(cities)
        end = time.time()

        time_taken = end - start
        times.append((i, time_taken))

    return times


def extrapolate_runtime(times, n, method: Literal["hill_climb", "exhaustive_search"]):
    """Chooses which function to extrapolate"""
    if method == "hill_climb":
        extrapolated_times, predict = extrapolate_hill(times, n)  # linear
    elif method == "exhaustive_search":
        extrapolated_times, predict = extrapolate_exhaustive(times, n)  # log-log factorial
    else:
        raise ValueError("Unknown method")

    return extrapolated_times, predict


def extrapolate_exhaustive(times, n):
    """Extrapolates time taken for n cities based on measured times
        Uses linear regression on log-log scale
        Created with the help of ChatGPT 4 (my idea, gpts implementation)

        Returns extrapolated times and a function to predict any n
    """
    n_vals = np.array([t[0] for t in times])
    t_vals = np.array([t[1] for t in times])

    # remove zero times because of log-log
    mask = t_vals > 0
    n_vals = n_vals[mask]
    t_vals = t_vals[mask]

    # log-log scale
    log_nfact = np.log([math.factorial(n) for n in n_vals])
    log_t = np.log(t_vals)

    coeff = np.polyfit(log_nfact, log_t, 1)
    slope, intercept = coeff

    print(f"\n === Extrapolation Model ===")
    print(f"Fitted model: log(T) = {slope:.3f} * log(n!) + {intercept:.3f}")

    # extrapolation
    extrapolate_times = []
    for i in range(n_vals[-1] + 1, n + 1):
        log_t_n = slope * math.log(math.factorial(i)) + intercept  # <-- uses extrapolation function with log(n!)
        t_n = np.exp(log_t_n)
        extrapolate_times.append((i, t_n))

    def predict(n):
        return np.exp(slope * math.log(math.factorial(n)) + intercept)

    return extrapolate_times, predict


def extrapolate_hill(times, n):
    """Extrapolates time taken for n cities based on measured times
        Uses a linear fit.
        Created with the help of ChatGPT 4 (my idea, gpts implementation)

        Linear fits best, as growth is not as fast as in factorial time, but roughly n*(n-1)/2 per step

       Returns extrapolated times and a prediction function.
    """
    n_vals = np.array([t[0] for t in times])
    t_vals = np.array([t[1] for t in times])

    # Linear regression: T ~ a*n + b
    coeff = np.polyfit(n_vals, t_vals, 1)
    a, b = coeff

    print(f"\n === Hill Climb Linear Extrapolation Model ===")
    print(f"Fitted model: T(n) = {a:.6f}*n + {b:.6f}")

    # extrapolated times
    extrapolated_times = []
    for i in range(n_vals[-1] + 1, n + 1):
        t_i = a * i + b
        extrapolated_times.append((i, t_i))

    # prediction function
    def predict(n):
        return a * n + b

    return extrapolated_times, predict
