import numpy as np
import matplotlib.pyplot as plt

voc = np.linspace(0.001, 0.5, 1000)

for i in voc:
    I0 = 2/(np.exp(28.99*i)-1)
    for j in voc:
        if j == round(i-np.log(1+28.99*j)/28.99,3):
            pm = j*(I0*(np.exp(28.99*j)-1)-2)
            plt.plot(j, pm, 'o', label = 'point')
plt.grid()
plt.xlabel(r'Voltage[V]')
plt.ylabel(r'Power[W]')
plt.legend(loc = 'best')
plt.show()