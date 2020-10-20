import numpy as np
import matplotlib.pyplot as plt

"""
Plotagem de uma reta simples f(x) = 2x
"""
arr1 = np.array([1, 2, 3, 4, 5, 6, 7 ,8, 9, 10])
arr2 = np.array([2, 4, 6, 8, 10, 12, 14 ,16, 18, 20])

plt.figure(figsize=(6, 4))
plt.plot(arr1, arr2)
plt.savefig('reta-simples.png')

plt.show()

