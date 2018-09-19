import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

filename = 'C:/Users/98415/Data_Analysis/raw_data.txt'
[distance, mean_time] = np.loadtxt(filename, delimiter=' ', unpack=True)
print(distance)
print(mean_time)

y = distance   #S=(0.5t^2)*g
x = 0.5*mean_time**2

slope, intercept, r_value, p_value, std_err = linregress(x,y)
#Fit x and y Data using a linear regression

dl=0 
dt=0

print('Slope, Error, Intercept, Correlation Coefficient:')
print(slope, std_err, intercept, r_value**2)
print('g = {} +- {} m/s^2'.format(slope, std_err))

xfit = np.linspace(0.0375,0.05,100)
yfit = slope*xfit + intercept

plt.plot(xfit, yfit, 'r-', label= 'Linear Fit')
plt.errorbar(x,y,fmt='o',label='Data', xerr=dl, yerr=dt) 
plt.grid()
plt.xlabel('distance[m]')
plt.ylabel('0.5*mean_time**2[s^2] ')
plt.legend(loc='best')

plt.show()
