import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
from scipy.stats import linregress

#this module could be used for two columns
#splitting into two linear regression is supported

def main():
    current_path_of_py = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(current_path_of_py)

    filename = input('Please input name of your worksheet(with type): ')
    worksheet = pd.read_excel(filename)
    df = pd.DataFrame(worksheet)

    breaking_point = input('Do you need Breaking point?(yes or no):')
    
    if breaking_point == 'yes':
        breaking_point = eval(input('Which one?(e.g.15. This means you wanna split data points into 0-14 and 14 to after):'))
   
        x_name = df.columns[0]
        y_name = df.columns[1]
        x_values = list(df[x_name].values)
        y_values = list(df[y_name].values)
        slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = linregress(x_values[0:breaking_point], y_values[0:breaking_point])
        slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = linregress(x_values[breaking_point:-1], y_values[breaking_point:-1])
        #Fit x and y Data using a linear regression
        xfit_1 = np.linspace(x_values[0], x_values[-1], 100)
        xfit_2 = np.linspace(x_values[0], x_values[-1], 100)
        cross_zero = input('Do you need intercept to be zero(yes or no):')
        if cross_zero == 'yes':
            intercept = 0
        else:
            pass
        yfit_1 = slope_1*xfit_1 + intercept_1
        yfit_2 = slope_2*xfit_2 + intercept_2
        plt.plot(xfit_1, yfit_1, 'r-', label= 'Linear Fit_1')
        plt.plot(xfit_2, yfit_2, 'g-', label= 'Linear Fit_2')
        plt.errorbar(x_values[0:breaking_point],y_values[0:breaking_point],fmt='o',label='Data point_1', xerr = 0, yerr = std_err_1, capsize = 5)
        plt.errorbar(x_values[breaking_point:-1],y_values[breaking_point:-1],fmt='o',label='Data point_2', xerr = 0, yerr = std_err_2, capsize = 5)
        '''
        we don't want error on x axis
        to make vertical line longer, we need to set parameter 'capsize' with appropriate value
        '''
        intersection_x = (intercept_2 - intercept_1)/(slope_1 - slope_2)
        intersection_y = (slope_1 * intersection_x + intercept_1)
        inters_y_mdf = intersection_y * 0.996
        plt.plot(intersection_x, intersection_y, 'bo', label = 'Intersection')
        plt.text(intersection_x, inters_y_mdf, '(' + str(round(intersection_x,2)) + ',' + str(round(intersection_y,2)) + ')')
        
        Unit_x = input('Please input unit of X axis(e.g.[m]):')
        Unit_y = input('Please input unit of Y axis(e.g.[m]):')
        print('-----------------------------------\n')
        print('Slope_1, Error_1, Intercept_1, Correlation Coefficient_1:') 
        # r^2 = coefficient of determination
        print(slope_1, std_err_1, intercept_1, r_value_1**2)
        print('-----------------------------------\n')
        print('Slope_2, Error_2, Intercept_2, Correlation Coefficient_2:')
        print(slope_2, std_err_2, intercept_2, r_value_2**2)
        print('-----------------------------------')
        
        plt.grid()
        plt.xlabel(x_name + Unit_x)
        plt.ylabel(y_name + Unit_y)
        plt.legend(loc = 'best')
        plt.show()
        
    elif breaking_point == 'no':
        x_name = df.columns[0]
        y_name = df.columns[1]
        x_values = list(df[x_name].values)
        y_values = list(df[y_name].values)
        slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = linregress(x_values, y_values)
        xfit_1 = np.linspace(x_values[0], x_values[-1], 100)
        cross_zero = input('Do you need intercept to be zero(yes or no):')
        if cross_zero == 'yes':
            intercept = 0
        else:
            pass
        yfit_1 = slope_1*xfit_1 + intercept_1
        plt.plot(xfit_1, yfit_1, 'r-', label= 'Linear Fit_1')
        plt.errorbar(x_values, y_values, fmt='o',label='Data point_1', xerr = 0, yerr = std_err_1, capsize = 5)
        # we don't want error on x axis
        # to make vertical line longer, we need to set parameter 'capsize' with appropriate value
        Unit_x = input('Please input unit of X axis(e.g.[m]):')
        Unit_y = input('Please input unit of Y axis(e.g.[m]):')
        print('-----------------------------------\n')
        print('Slope_1, Error_1, Intercept_1, Correlation Coefficient_1:') 
        # r^2 = coefficient of determination
        print(slope_1, std_err_1, intercept_1, r_value_1**2)
        print('-----------------------------------\n')
        
        plt.grid()
        plt.xlabel(x_name + Unit_x)
        plt.ylabel(y_name + Unit_y)
        plt.legend(loc = 'best')
        plt.show()
        
    else:
        print('Invalid input')
        
   
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
