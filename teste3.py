
"""

import numpy as np
import matplotlib.pyplot as plt


#Mais de uma linha no gráfico

x = np.arange(1, 11)
print(x)

a1 = [1, 2, 3, 4]
a2 = [2, 4, 6, 8]

a3 = [5, 6, 7, 8]
a4 = [10, 12, 14, 16]


plt.figure(figsize=(6, 4))
plt.plot(a1, a2)
plt.plot(a3, a4)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
#plt.savefig('reta-simples-duas.png')

plt.show()
"""

"""

import numpy as np
import matplotlib.pyplot as plt


#Mais de uma linha no gráfico

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
"""


import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

data = np.random.rand(8, 8)
ax = sns.heatmap(data, linewidth=0.3)
plt.show()

print(data)

