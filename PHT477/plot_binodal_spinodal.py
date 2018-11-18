import math
import numpy as np
import matplotlib.pyplot as plt

phi = np.linspace(0.05, 0.95, 200)
f = lambda x:(np.log(1-x)-np.log(x))/(1-2*x)
chi_bi = f(phi)
# plt.subplot(221)
plt.plot(phi, chi_bi, 'r-', label= 'binodal interaction')
g = lambda y:1/(2*y*(1-y))
chi_sp = g(phi)
plt.plot(phi, chi_sp, 'b-', label= 'spinodal interaction')
plt.grid()
plt.xlabel('phi')
plt.ylabel('chi')
plt.legend(loc = 'best')
plt.show()
'''
g = lambda y:1/(2*y*(1-y))
chi_sp = g(phi)
plt.subplot(224)
plt.plot(phi, chi_sp, 'b-', label= 'spinodal interaction')
plt.grid()
plt.xlabel('phi')
plt.ylabel('chi')
plt.legend(loc = 'best')
plt.show()
'''