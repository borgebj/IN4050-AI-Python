import matplotlib.pyplot as plt
from utils import format_time, measure_runtime, extrapolate_runtime
from utils import city_names

# Map of Europe
europe_map = plt.imread('map.png')

# Lists of city coordinates
city_coords = {
    "Barcelona": [2.154007, 41.390205], "Belgrade": [20.46, 44.79], "Berlin": [13.40, 52.52],
    "Brussels": [4.35, 50.85], "Bucharest": [26.10, 44.44], "Budapest": [19.04, 47.50],
    "Copenhagen": [12.57, 55.68], "Dublin": [-6.27, 53.35], "Hamburg": [9.99, 53.55],
    "Istanbul": [28.98, 41.02], "Kyiv": [30.52, 50.45], "London": [-0.12, 51.51],
    "Madrid": [-3.70, 40.42], "Milan": [9.19, 45.46], "Moscow": [37.62, 55.75],
    "Munich": [11.58, 48.14], "Paris": [2.35, 48.86], "Prague": [14.42, 50.07],
    "Rome": [12.50, 41.90], "Saint Petersburg": [30.31, 59.94], "Sofia": [23.32, 42.70],
    "Stockholm": [18.06, 60.33], "Vienna": [16.36, 48.21], "Warsaw": [21.02, 52.24]}


def plot_plan(city_order):
    """Plots given plan (list of city names) on the map."""
    assert city_order is not None

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(europe_map, extent=[-14.56, 38.43, 37.697 + 0.3, 64.344 + 2.0], aspect="auto")

    # Map (long, lat) to (x, y) for plotting
    for index in range(len(city_order) - 1):
        current_city_coords = city_coords[city_order[index]]
        next_city_coords = city_coords[city_order[index + 1]]
        x, y = current_city_coords[0], current_city_coords[1]

        # Plotting a line to the next city
        next_x, next_y = next_city_coords[0], next_city_coords[1]
        ax.plot([x, next_x], [y, next_y])

        ax.plot(x, y, 'ok', markersize=5)
        ax.text(x, y, index, fontsize=12)

    # Finally, plotting from last to first city
    first_city_coords = city_coords[city_order[0]]
    first_x, first_y = first_city_coords[0], first_city_coords[1]
    ax.plot([next_x, first_x], [next_y, first_y])

    # Plotting a marker and index for the final city
    ax.plot(next_x, next_y, 'ok', markersize=5)
    ax.text(next_x, next_y, index + 1, fontsize=12)

    return fig, ax


def plot_times(measured, extrapolated, function):
    """Plots measured and extrapolated times on a graph"""
    fig, ax = plt.subplots()

    measured_x = [t[0] for t in measured]
    measured_y = [t[1] for t in measured]
    extrapolated_x = [t[0] for t in extrapolated]
    extrapolated_y = [t[1] for t in extrapolated]

    ax.plot(measured_x, measured_y, 'bo', label='Measured')
    ax.plot(extrapolated_x, extrapolated_y, 'r--', label='Extrapolated')

    ax.set_yscale('log')
    ax.set_xlabel('Number of cities (n)')
    ax.set_ylabel('Time (seconds, log scale)')
    fun_name = function.__name__.replace('_', ' ').title()
    ax.set_title(f'{fun_name} Runtime: Measured vs Extrapolated')
    ax.legend()
    ax.grid(True, which='both', ls='--')

    return fig, ax


def plotter(LIMIT, MAX_EXTRAPOLATE, path, function, extrapolate=True):
    """Does the actual plotting + optional extrapolation"""

    if extrapolate:
        # 1. Measure
        times_measured = measure_runtime(function, LIMIT, step=1)

        # 2. Extrapolate
        function_name = function.__name__
        times_extrapolated, predict = extrapolate_runtime(times_measured, MAX_EXTRAPOLATE, function_name)

        print("\nPredicted times for values:\n")
        for n in [5, 10, 15, 20, 24, 28, 32, 36, 40, 44, 48]:
            t_sec = float(predict(n))
            print(f"{n:2d} cities: {format_time(t_sec)}")

        # 3. Plot measured + extrapolated times
        plot_times(times_measured, times_extrapolated, function)

    # plot plan, picture of path
    plot_plan(path)
    show_all_figures()


def show_all_figures():
    plt.show()


if __name__ == "__main__":
    # Example usage of the plotting-method.
    plan = list(city_coords.keys())  # Gives us the cities in alphabetic order
    print(plan)
    plot_plan(plan)
    show_all_figures()
