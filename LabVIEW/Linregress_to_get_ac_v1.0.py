import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

filelist = ['U:/Data_git/Data_Analysis/LabVIEW/IR LED-ten films- without shelter.txt',\
'U:/Data_git/Data_Analysis/LabVIEW/Red LED- ten films- without shelter.txt']

num = int(input('Which one (choose from 1 or 2)? = '))
filename = filelist[num-1]
[LHS, RHS] = np.loadtxt(filename, delimiter=',', unpack = True)
# LHS = ln(i/i0), RHS = d

y = LHS  
x = RHS
slope, intercept, r_value, p_value, std_err = linregress(x,y)
#Fit x and y Data using a linear regression
print(std_err)

a = RHS[-1]
xfit = np.linspace(0, a, 100)
intercept = 0 # we set intercept to zero as we need
yfit = slope*xfit + intercept
plt.plot(xfit, yfit, 'r-', label= 'Linear Fit')
plt.errorbar(x,y,fmt='o',label='Data point', xerr = 0, yerr = std_err, capsize = 3) 
# we don't want error on x axis
# to make horizontal line longer, we need to set parameter 'capsize' with appropriate value
plt.grid()
plt.ylabel('ln(I/I_ZERO)[1]')
plt.xlabel('d[mm]')
plt.legend(loc = 'best')
print('Slope, Error, Intercept, Correlation Coefficient:') # r^2 = coefficient of determination
print(slope, std_err, intercept, r_value**2)
plt.show()
