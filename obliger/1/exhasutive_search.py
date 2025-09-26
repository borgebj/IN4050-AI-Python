import csv
from itertools import permutations
from numba import jit, cuda
import time

with open("european_cities.csv", "r") as f:
    data = list(csv.reader(f, delimiter=';'))
    cities = data[0]


def data_to_dict(data: list) -> dict:
    """Converts the csv data to a dictionary of dictionaries."""
    None


def get_distance(city1: str, city2: str, data: list) -> int:
    """Returns the distance between two cities."""
    None


@jit(target_backend='cuda')
def permute(cities: list) -> list:
    """Returns all permutations of the given list of cities."""
    return list(permutations(cities))


limit = 11
cities = cities[:limit]

start = time.time()
city_perms = permute(cities)
end = time.time()

print(f"Cities + distances: ")
print(*data, sep="\n")
print("\nCities: ")
print(cities)

print("\nPermutations of cities: ")
print(city_perms)
print(f"\nTime taken to compute {len(city_perms)} permutations of {limit} cities: {end - start:.4f} seconds")
