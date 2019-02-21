import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

filename = input('Please input name of your worksheet: ')
worksheet = pd.read_excel('filename')
df = pd.DataFrame(worksheet)

def paint():
	name_x = df.iloc[1:2, 1:2]
	x = df[name_x]
	name_y = df.iloc[1:2, ]

	
	
[LHS, RHS] = np.loadtxt(filename, delimiter=',', unpack = True) # LHS = h, RHS = Vx^2

y = LHS  
x = RHS
slope, intercept, r_value, p_value, std_err = linregress(x,y)
#Fit x and y Data using a linear regression

a = RHS[-1]
xfit = np.linspace(0, a, 100)
intercept = 0 # we set intercept to zero as we need
yfit = slope*xfit + intercept
plt.plot(xfit, yfit, 'r-', label= 'Linear Fit')
plt.errorbar(x,y,fmt='o',label='Data point', xerr = 0, yerr = std_err, capsize = 3) 
# we don't want error on x axis
# to make horizontal line longer, we need to set parameter 'capsize' with appropriate value
plt.grid()
plt.ylabel('h[m]')
plt.xlabel('Vx^2[m^2/s^2]')
plt.legend(loc = 'best')
print('Slope, Error, Intercept, Correlation Coefficient:') # r^2 = coefficient of determination
print(slope, std_err, intercept, r_value**2)
if num == 1: 
    print('standard I in 2-1 = {}'.format(0.001550846))
    I_exp = 2*0.051**2*(1.1925*9.8*slope-1.1925/2)
    print('I in exp = {}'.format(I_exp))
    plus = 2*0.051**2*(1.1925*9.8*(slope + std_err)-1.1925/2)
    print('Error of I in exp = {}'.format(plus - I_exp))
    minus = 2*0.051**2*(1.1925*9.8*(slope - std_err)-1.1925/2)
    print('Error of I in exp = {}'.format(I_exp - minus))
    print('I = {} +- {}'.format(I_exp, plus - I_exp))
elif num == 2: 
    print('standard I in 2-2 = {}'.format(0.000586376))
    I_exp = 2*0.0335**2*(1.045*9.8*slope-1.045/2)
    print('I in exp = {}'.format(I_exp)) 
    plus = 2*0.0335**2*(1.045*9.8*(slope + std_err)-1.045/2)
    print('Error of I in exp = {}'.format(plus - I_exp))
    minus = 2*0.0335**2*(1.045*9.8*(slope - std_err)-1.045/2)
    print('Error of I in exp = {}'.format(I_exp - minus))
    print('I = {} +- {}'.format(I_exp, plus - I_exp))
elif num == 3:
    print('standard I in 2-3 = {}'.format(0.003133555))
    I_exp = 2*0.051**2*(2.4095*9.8*slope-2.4095/2)
    print('I in exp = {}'.format(I_exp))
    plus = 2*0.051**2*(2.4095*9.8*(slope + std_err)-2.4095/2)
    print('Error of I in exp = {}'.format(plus - I_exp))
    minus = 2*0.051**2*(2.4095*9.8*(slope - std_err)-2.4095/2)
    print('Error of I in exp = {}'.format(I_exp - minus))
    print('I = {} +- {}'.format(I_exp, plus - I_exp))
plt.show()
