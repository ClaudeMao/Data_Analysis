import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit #another way to fit curve

#filename = 'C:/Users/98415/Data_Analysis/Experiments/raw_data.txt'
filename = 'U:/Data_git/Data_Analysis/Experiments/raw_data.txt'
[distance, mean_time] = np.loadtxt(filename, delimiter=' ', unpack=True)
print(distance)
print(mean_time)

y = distance   #S=(0.5t^2)*g, so value of slope would be g that we want
x = 0.5*mean_time**2

slope, intercept, r_value, p_value, std_err = linregress(x,y)
#Fit x and y Data using a linear regression

#errorname = 'C:/Users/98415/Data_Analysis/Experiments/error_data.txt'
errorname = 'U:/Data_git/Data_Analysis/Experiments/error_data.txt'
[dt, dl] = np.loadtxt(errorname, delimiter=' ', unpack=True)
dt = 0.5*dt**2

print(std_err)
print('Slope, Error, Intercept, Correlation Coefficient:')
print(slope, std_err, intercept, r_value**2)
print('g = {} +- {} m/s^2'.format(slope, std_err))

xfit = np.linspace(0.038,0.044,100)
yfit = slope*xfit + intercept

plt.plot(xfit, yfit, 'r-', label= 'Linear Fit')
plt.errorbar(x,y,fmt='o',label='Data', xerr=dt, yerr=dl) 
plt.grid()
plt.ylabel('distance[m]')
plt.xlabel('0.5*mean_time**2[s^2] ')
plt.legend(loc='best')

plt.show()