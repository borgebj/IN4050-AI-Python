from utils import city_names, path_distance, format_time
from plotter import plot_plan
import random
import time


# <----------- MAIN OPTIMIZATION FUNCTIONS ----------> #
def generate_start(cities):
    """Takes a list of cities and shuffles it"""
    random.shuffle(cities)
    return cities


def generate_neighbors(path):
    """
    Generates neighboring paths by swapping two cities in the current path.
    
    cities: list of city indices
    """

    neighbors = []
    cities = len(path)

    # for every city
    for i in range(cities):
        # swap with every other city after
        for j in range(i + 1, cities):
            neighbor = path.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # swap
            neighbors.append(neighbor)

    return neighbors


def hill_climb(cities, verbose=False):
    """
    Finds the shortest path among the (one) given permutations of cities generated at start.

    cities: list of city indices
    """

    # chooses an arbitrary (random) start, as well as its distance
    start = generate_start(cities)
    current_shortest = path_distance(start)

    step = 0  # step counter
    while True:
        neighbors = generate_neighbors(start)

        # find the smallest value of neighbors based on 'get_path_distance'
        best_neighbor = min(neighbors, key=path_distance)
        best_distance = path_distance(best_neighbor)

        # display in terminal
        if verbose:
            start_names = [city_names[i] for i in start]
            best_names = [city_names[i] for i in best_neighbor]
            print(f"\n[{step}] Start: {start_names} ({current_shortest:.2f})")
            print(f"[{step}] Best:  {best_names} ({best_distance:.2f})\n")

        # if shorter path exists: continue, else STOP
        if best_distance < current_shortest:
            start = best_neighbor
            current_shortest = best_distance
            step += 1
        else:
            break

    return start, current_shortest, step


def run_statistics(function, cities):
    """
    Does 20 runs, prints worst and mean distances, plots the middle run
    """
    runs = 20
    distances = []
    best_distance = float('inf')

    # does 20 runs, saves distances
    for i in range(runs):
        path, distance, step = function(cities)
        distances.append(distance)

        if distance < best_distance:
            best_distance = distance

        # plot the middle
        if i == runs // 2:
            # turn indices back to string
            path_names = [city_names[i] for i in path]
            plot_plan(path_names)

    worst = max(distances)
    mean = sum(distances) / runs

    print(f"Worst distance over {runs} runs: {worst:.4f}")
    print(f"Mean distance over {runs} runs:  {mean:.4f}")
    print(f"Best distance over {runs} runs: {best_distance:.4f}")


def main():
    # main flags
    verbose = False
    LIMIT = 24  # 24 max

    # limits no. cities
    cities = list(range(LIMIT))  # represents cities as indexes

    # finds the shortest path in regard to neighboring paths
    start = time.time()
    path, distance, step = hill_climb(cities, verbose=verbose)
    end = time.time()

    # prints info on main run
    path_names = [city_names[i] for i in path]
    path_str = ' -> '.join(path_names) + f" -> {path_names[0]}"
    print("\n=== Results ===\n")
    print(f"Shortest path:\n    {path_str}\n")
    print(f"Distance:\n    {distance:.4f}\n")
    print(f"Total neighbors visited:\n    {step + 1}\n")
    print(f"Time taken for {LIMIT} cities:\n    {format_time(end - start)}\n")

    # statistics (worst, mean) + plot
    run_statistics(hill_climb, cities)


if __name__ == "__main__":
    main()
