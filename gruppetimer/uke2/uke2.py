
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return -np.power(x, 4) + 2 * np.power(x, 3) + 2 * np.power(x, 2) - x

def df(x):
    return -4 * np.power(x, 3) + 6 * np.power(x, 2) + 4 * x - 1


# plotting from x = -2 to x = 3
x = np.linspace(-2, 3, 400)


y = f(x)
dy = df(x)

plt.plot(x, y, label="f(x)")
plt.plot(x, dy, label="f'(x)", linestyle="--")

plt.title("Function f(x) and its derivatie f'(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)
plt.legend()
plt.grid(True)
plt.show()

# Oppgave 2

# Gradient ascent
gamma = 0.1         # step size
x = 0               # starting point (any value in [-2, 3]
iterations = 20     # no. updates

x_values = [x]
y_values = [f(x)]

for i in range(iterations):
    x = x + gamma * df(x)   # update rule
    x_values.append(x)
    y_values.append(f(x))

# Plot function
xx = np.linspace(-2, 3, 400)
yy = f(xx)

plt.plot(xx, yy, label="f(x)")
plt.scatter(x_values, y_values, color="red", zorder=5, label="Gradient Ascent Path")
plt.plot(x_values, y_values, color="red", linestyle="--", alpha=0.6)

plt.title("Gradient Ascent on f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
