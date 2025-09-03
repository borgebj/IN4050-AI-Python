import numpy as np
import time as t

# list
mylist = list(range(1,10**7))
start = t.time()

sum = 0
for e in mylist:
    sum += 1/e**2

print(f"Sum: {sum:.5f}, in {t.time() - start:.2f} s")

# array
myarr = np.arange(1, 10**7)
start = t.time()

sum = np.sum(1/myarr**2)

print(f"Sum: {sum:.5f}, in {t.time() - start:.2f} s")

# First column
first_col = [row[0] for row in list]
