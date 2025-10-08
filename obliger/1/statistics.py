import time
import numpy as np
import math
from plotter import plot_plan, plot_times
from utils import city_names, format_time


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
        (i, predict(i)) for i in range(max(x) + 1, n + 1)
    ]

    return extrapolate_times, predict


def plot_extrapolation(limit, max_extrapolate, function, extrapolate=True):
    """Does the actual plotting + optional extrapolation"""

    if extrapolate:
        # 1. Measure up to limit
        times_measured = measure_runtime(function, limit)

        # 2. Extrapolate
        times_extrapolated, predict = extrapolate_exhaustive(times_measured, max_extrapolate)

        print("\nPredicted times for values:\n")
        for n in [5, 10, 15, 20, 24, 28, 32, 36, 40, 44, 48]:
            t_sec = float(predict(n))
            print(f"{n:2d} cities: {format_time(t_sec)}")

        # 3. Plot measured + extrapolated times
        plot_times(times_measured, times_extrapolated, function)


def run_statistics(function):
    """
    Does 20 runs, prints worst and mean distances, plots the middle run
    """
    runs = 20

    print("\n=== Statistics ===")

    # runs hill with 10 and 24 cities
    for limit in [10, 24]:
        cities = list(range(limit))

        best_distance = float('inf')
        best_path = None
        distances = []

        # run 20 times
        for run in range(runs):
            path, distance, _ = function(cities)
            distances.append(distance)

            # get best
            if distance < best_distance:
                best_distance = distance
                best_path = path

        # plot best  - will plot twice, for 20 and 24
        path_names = [city_names[i] for i in best_path]
        plot_plan(path_names)

        # calculate statistics
        worst = max(distances)
        mean = sum(distances) / runs

        # square each deviation
        deviations = [((distance - mean) ** 2) for distance in distances]
        variance = sum(deviations) / (len(distances) - 1)
        standard_deviation = variance ** 0.5

        # display statistics
        print(f"\n== {limit} cities ==")
        print(f"Worst distance over {runs} runs: {worst:15.4f}")
        print(f"Mean  distance over {runs} runs: {mean:15.4f}")
        print(f"Best  distance over {runs} runs: {best_distance:15.4f}")
        print(f"Standard deviation over {runs} runs: {standard_deviation:11.4f}")
