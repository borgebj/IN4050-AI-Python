from argparse import ArgumentParser
import csv

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


def parse_args():
    """Command line arguments."""
    parser = ArgumentParser(description="CLI arguments for TSP algorithms")

    parser.add_argument(
        "-l", "--limit", "--cities",
        type=int,
        required=True,
        help="Number of cities in path (max 24)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose output (may be very long)"
    )

    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (used in GA)"
    )

    args = parser.parse_args()
    return args


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
