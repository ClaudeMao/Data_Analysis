import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

filelist = ['U:/Data_git/Data_Analysis/Experiments/Rolling_motion/2-1.txt',\
'U:/Data_git/Data_Analysis/Experiments/Rolling_motion/2-2.txt',\
'U:/Data_git/Data_Analysis/Experiments/Rolling_motion/2-3.txt']

num = int(input('Which one (choose from 1,2,3)?:'))
filename = filelist[num-1]
[LHS, RHS] = np.loadtxt(filename, delimiter=',', unpack = True) # LHS = h, RHS = Vx^2

y = LHS  
x = RHS
slope, intercept, r_value, p_value, std_err = linregress(x,y)
#Fit x and y Data using a linear regression
'''
#errorname = 'C:/Users/98415/Data_Analysis/Experiments/error_data.txt'
errorname = 'U:/Data_git/Data_Analysis/Experiments/error_data.txt'
[dt, dl] = np.loadtxt(errorname, delimiter=' ', unpack=True)
dt = 0.5*dt**2

print(std_err)
print('Slope, Error, Intercept, Correlation Coefficient:')
print(slope, std_err, intercept, r_value**2)
print('g = {} +- {} m/s^2'.format(slope, std_err))
'''

a = RHS[-1]
xfit = np.linspace(0, a, 100)
yfit = slope*xfit + intercept
plt.plot(xfit, yfit, 'r-', label= 'Linear Fit')
plt.errorbar(x,y,fmt='o',label='Data', xerr=0, yerr=0) 
plt.grid()
plt.ylabel('h[m]')
plt.xlabel('Vx^2[m^2/s^2]')
plt.legend(loc='best')
print('Slope, Error, Intercept, Correlation Coefficient:')
print(slope, std_err, intercept, r_value**2)
if num == 1:
    print('standard I in 2-1:{}'.format(0.001550846))
    print('I in exp:{}'.format(2*0.051**2*(1.1925*9.8*slope-1.1925/2)))
elif num == 2:
    print('standard I in 2-2:{}'.format(0.000586376))
    print('I in exp:{}'.format(2*0.0335**2*(1.045*9.8*slope-1.045/2)))
elif num == 3:
    print('standard I in 2-3:{}'.format(0.003133555))
    print('I in exp:{}'.format(2*0.051**2*(2.4095*9.8*slope-2.4095/2)))
plt.show()
