import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

filename = 'C:/Users/98415/Data_Analysis/example.txt'
[length,time] = np.loadtxt(filename, delimiter=',',unpack=True)
print(length) 
print(time)

dl=2e-3 
dt=0.005 
x = np.sqrt(length) 
y = time
slope, intercept, r_value, p_value, std_err = linregress(x,y)
g= 4*np.pi**2/slope**2 
dg= 8*np.pi**2*(std_err/slope**3)
print('Slope, Error, Intercept, Correlation Coefficient:' )
print(slope, std_err, intercept, r_value**2 )
print('g=',g,'+/',dg,'m/s')

xfit = np.linspace(0, 0.8, 100) 
yfit = slope*xfit + intercept
plt.plot(xfit,yfit, 'r-', label='Linear Fit') 
plt.errorbar(np.sqrt(length),time,fmt='o',label='Data', xerr=dl, yerr=dt)
plt.xlabel('$\sqrt{L}$ [m$^{1/2}$]') 
plt.ylabel('Period [s] ')
plt.legend(loc='best')
plt.show()




