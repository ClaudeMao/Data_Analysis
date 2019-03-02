import numpy as np
import pandas as pd
import os, sys, math
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import chisquare
from scipy.optimize import curve_fit

#this module could be used for two columns
#gaussfit: 3 peaks are supported


def one_gaussian(x,A,B,C): #Note: had better not to use variable parameter
    return A*np.exp(-(x-B)**2 / (2*C**2))

'''
def gaussian(x,*param): # param should be a list or tuple, condition of 4 peaks are supported
    return param[0] * np.exp(-(x - param[1])**2 / (2 * param[2]**2)) + \
           param[3] * np.exp(-(x - param[4])**2 / (2 * param[5]**2)) + \
           param[6] * np.exp(-(x - param[7])**2 / (2 * param[8]**2)) + \
           param[9] * np.exp(-(x - param[10])**2 / (2 * param[11]**2))
'''

def two_gaussians(x, A1, A2, B1, B2, C1, C2):
    return (one_gaussian(x, A1, B1, C1) + one_gaussian(x, A2, B2, C2))
    
def three_gaussians(x, A1, A2, A3, B1, B2, B3, C1, C2, C3):
    return (one_gaussian(x, A1, B1, C1) + one_gaussian(x, A2, B2, C2) + one_gaussian(x, A3, B3, C3))

def FWHM(sd):
    return 2*np.sqrt(2*np.log(2))*sd

def main():
    current_path_of_py = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(current_path_of_py)

    filename = input('Please input name of your worksheet(with type): ')
    worksheet = pd.read_excel(filename)
    df = pd.DataFrame(worksheet)


    x_name = df.columns[0]
    y_name = df.columns[1]
    x_values = list(df[x_name].values)
    y_values = list(df[y_name].values)
    
    print('Instruction: 1 for one gaussian fit, 2 for two gaussian fit, 3 for three gaussian fit!')
    hm_gauss = eval(input('How many peaks do you need to plot(please input interger from 1-3):'))
    print('Instruction: p1 = [5000,675,150] p2 = [5000,4650,650,750,10,10] p3 = [15000,5000,4650,400,650,750,10,100,100]')
    p0 = []
    for i in range(hm_gauss*3):
        parameter = eval(input('Please input your guesswork for A1 A2 A3 B1 B2 B3 C1 C2 C3:'))
        p0.append(parameter)
        
        
    if hm_gauss == 1:
        gauss_num = 1
        hm_gauss = one_gaussian
    elif hm_gauss == 2:
        gauss_num = 2
        hm_gauss = two_gaussians
    elif hm_gauss == 3:
        gauss_num = 3
        hm_gauss = three_gaussians
    
    
    popt, pcov = curve_fit(hm_gauss, x_values, y_values, p0)  #maybe sigma is needed
    xfit = np.linspace(x_values[0], x_values[-1], 5000)
    yfit = hm_gauss(xfit, *popt)# popt[3],popt[4],popt[5])
    plt.errorbar(x_values, y_values, lw = 1, label='Data')
    plt.plot(xfit, yfit, lw = 2, label='GaussFit')
    plt.legend(loc = 'best')

    
    # Labels
    Unit_x = input('Please input unit of X axis(e.g.[m]):')
    Unit_y = input('Please input unit of Y axis(e.g.[m]):')
    plt.xlabel(x_name + Unit_x)
    plt.ylabel(y_name + Unit_y)
    
    # Print the optimised fit parameters 
    if gauss_num == 1:
        print('First Gaussfit-----Peak:{}W/mm2\n\
                       Centred at:{}mm\n\
                       Standard Deviation:{}mm'.format(popt[0],popt[1],popt[2]))
    if gauss_num == 2:
        print('First Gaussfit-----Peak:{}W/mm2\n\
                       Centred at:{}mm\n\
                       Standard Deviation:{}mm'.format(popt[0],popt[1],popt[2]))
        print('Second Gaussfit-----Peak:{}W/mm2\n\
                        Centred at:{}mm\n\
                        Standard Deviation:{}mm'.format(popt[3],popt[4],popt[5]))
    if gauss_num == 3:
        print('First Gaussfit-----Peak:{}W/mm2\n\
                   Centred at:{}mm\n\
                   Standard Deviation:{}mm'.format(popt[0],popt[1],popt[2]))
        print('Second Gaussfit-----Peak:{}W/mm2\n\
                    Centred at:{}mm\n\
                    Standard Deviation:{}mm'.format(popt[3],popt[4],popt[5]))
        print('Third Gaussfit-----Peak:{}W/mm2\n\
                   Centred at:{}mm\n\
                   Standard Deviation:{}mm'.format(popt[6],popt[7],popt[8]))                   
    plt.show()
    
if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print('No such file found!')
    except ValueError as e:
        print('Values input should be less than range of index!')
    
    
    
    
    
    
    
    
    
