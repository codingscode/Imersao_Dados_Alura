
import numpy as np
import matplotlib.pyplot as plt

"""
Mais de uma linha no gráfico
"""
x = np.arange(1, 11)
print(x)

plt.figure(figsize=(6, 4))
plt.plot(x, 2*x)
plt.plot(x, x/2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
#plt.savefig('reta-simples-duas.png')

plt.show()




