import numpy as np
import pandas as pd
import os, sys, math
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import chisquare

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
        errorbar = df.columns[2]
        x_values = list(df[x_name].values)
        y_values = list(df[y_name].values)
        y_errorbar = list(df[errorbar].values)
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
        
        
        #calculate intersection and plot it on graph
        intersection_x = (intercept_2 - intercept_1)/(slope_1 - slope_2)
        intersection_y = (slope_1 * intersection_x + intercept_1)
        inters_y_mdf = intersection_y * 0.996
        plt.plot(intersection_x, intersection_y, 'bo', label = 'Intersection')
        plt.text(intersection_x, inters_y_mdf, '(' + str(round(intersection_x,2)) + ',' + str(round(intersection_y,2)) + ')')
        
        
        #input unit of axis
        Unit_x = input('Please input unit of X axis(e.g.[m]):')
        Unit_y = input('Please input unit of Y axis(e.g.[m]):')
        
        
        #print out result of each line
        print('-----------------------------------')
        print('Slope_1, Error_1, Intercept_1, Correlation Coefficient_1:') # r^2 = coefficient of determination
        print(slope_1, std_err_1, intercept_1, r_value_1**2)
        
        
        #calculate std_err_on_y of line 1
        def power_1(a):
            return a**slope_1 + intercept_1
        y_fit_values_1 = list(map(power_1,x_values[0:breaking_point])) #this is a list which only contain several fitted points
        y_y_fit_one = [y_values[0:breaking_point][i]-y_fit_values_1[i] for i in range(len(y_fit_values_1))] #Put 'y - y_fit' of line 1 into list
        summa_one = [i**2 for i in y_y_fit_one]
        total_one = 0
        for i in range(len(summa_one)):
            total_one += summa_one[i]
        std_err_on_y_1 = math.sqrt(total_one / (len(y_fit_values_1) - 2))
        
        
        #calculate chi-squared of line 1
        chi_squ_1 = 0
        for i in range(breaking_point):
            chi_squ_1 += ((y_values[0:breaking_point][i] - y_fit_values_1[i])/std_err_on_y_1)**2
        print('chi square test: {}'.format(round(chi_squ_1,2)))# two digits
        
        
        #calculate std_err_on_y of line 2
        def power_2(a):
            return a**slope_2 + intercept_2
        y_fit_values_2 = list(map(power_2,x_values[breaking_point:-1]))
        y_y_fit_two = [y_values[breaking_point:-1][i]-y_fit_values_2[i] for i in range(len(y_fit_values_2))] #Put 'y-y_fit' of line 2 into list
        summa_two = [i**2 for i in y_y_fit_two]
        total_two = 0
        for i in range(len(summa_two)):
            total_two += summa_two[i]
        std_err_on_y_2 = math.sqrt(total_two / (len(y_fit_values_2) - 2))

        print('-----------------------------------')
        print('Slope_2, Error_2, Intercept_2, Correlation Coefficient_2:')
        print(slope_2, std_err_2, intercept_2, r_value_2**2)
        
        
        #calculate chi-squared of line 2
        chi_squ_2 = 0
        for i in range(len(y_fit_values_2)):
            chi_squ_2 += ((y_values[breaking_point:-1][i] - y_fit_values_2[i])/std_err_on_y_2)**2
        print('chi square test: {}'.format(round(chi_squ_2,2))) # two digits
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
    
    
    
    
    
    
    
    
    
    
