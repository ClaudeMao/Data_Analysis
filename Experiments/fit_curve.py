import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
#from scipy.stats import linregress
from scipy.optimize import curve_fit

task = pd.read_excel('C:/Users/98415/Data_Analysis/raw_data.xlsx')
df = pd.DataFrame(task)

def f(x,a,b):
    return a*x+b

def paint():
    x = df.loc[['distance'],['case1','case2','case3','case4','case5']]
    y = df.loc[['T^2 1'],['case1','case2','case3','case4','case5']]
    plt.grid()
    plt.scatter(x, y, c='r', marker='.', label='case1')
    plt.title(u'Distance and time square')
    plt.legend(loc='upper right')
    plt.xlabel(u'time square')
    plt.ylabel(u'distance')
    popt, pcov = curve_fit(f, x, y)
    a = popt[0]
    b = popt[1]
    y_value = f(x,a,b)
    plot1 = plt.plot(x, y, 's',label='original values')
    plot2 = plt.plot(x, y_value, 'r',label='polyfit values')
    plt.show()

paint()