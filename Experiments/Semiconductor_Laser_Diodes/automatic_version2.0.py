import numpy as np
import pandas as pd
import os, sys, math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.stats import chisquare

#this module could be used for two columns
#splitting into two linear regression is supported
#gaussian fit is supported
#all input used should be small letter

def main():
    current_path_of_py = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(current_path_of_py)

    filename = input('Please input name of your worksheet(with type): ')
    worksheet = pd.read_excel(filename)
    df = pd.DataFrame(worksheet)
    
    VIP = eval(input('Important! Linear(0) or curve(1):'))
    if VIP == 0:
        breaking_point = eval(input('Do you need Breaking point? YES(1) or NO(0):'))
    elif VIP == 1:
        CurveType = input('Important! Gaussian(a) or exponetial(b):')
        if CurveType == 'a':
            def one_gaussian(x,A,B,C): #Note: had better not to use variable parameter
                return A*np.exp(-(x-B)**2 / (2*C**2))
                
            def two_gaussians(x, A1, A2, B1, B2, C1, C2):
                return (one_gaussian(x, A1, B1, C1) + one_gaussian(x, A2, B2, C2))
    
            def three_gaussians(x, A1, A2, A3, B1, B2, B3, C1, C2, C3):
                return (one_gaussian(x, A1, B1, C1) + one_gaussian(x, A2, B2, C2) + one_gaussian(x, A3, B3, C3))

            def FWHM(sd):
                return 2*np.sqrt(2*np.log(2))*sd
            
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
            
        if CurveType == 'b':
            def exponetial(x, a, b):
                return a * np.exp(x/b)
                
            x_name = df.columns[0]
            y_name = df.columns[1]
            x_values = list(df[x_name].values)
            y_values = list(df[y_name].values)
            
            popt, pcov = curve_fit(exponetial, x_values, y_values)  #maybe sigma is needed
            '''
            #calculate std_err_on_y of curve
            def exp(a):
                return popt[0]* np.exp(a/popt[1])
            y_fit_values = list(map(exp,x_values))
            y_y_fit = [y_values[i]-y_fit_values[i] for i in range(len(y_fit_values))] #Put 'y - y_fit' of curve into list
            summa_one = [i**2 for i in y_y_fit]
            total_one = 0
            for i in range(len(summa_one)):
                total_one += summa_one[i]
            std_err_on_y = math.sqrt(total_one / (len(y_fit_values) - 2))
            '''
            xfit = np.linspace(x_values[0], x_values[-1], 5000)
            yfit = exponetial(xfit, *popt)
            plt.errorbar(x_values, y_values, fmt = '.', label='Data')# , xerr = 0, yerr = std_err_on_y, capsize = 2)
            plt.plot(xfit, yfit, lw = 2, label='Exponetial fit')
            plt.legend(loc = 'best')
            # plt.text(x_values[-4], y_values[-4], 'y = ' + str(round(popt[0],2)) + '*exp(x/' + str(round(popt[1],2)) + ')')
            print('Values of a and b:{} , {}'.format(popt[0],popt[1]))

            # Labels
            Unit_x = input('Please input unit of X axis(e.g.[m]):')
            Unit_y = input('Please input unit of Y axis(e.g.[m]):')
            plt.xlabel(x_name + Unit_x)
            plt.ylabel(y_name + Unit_y)
            plt.show()
            
    # Below are Linear part!
    if breaking_point == 1: #where 1 means yes
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
        cross_zero = eval(input('Do you need intercept to be zero? YES(1) or NO(0):'))
        if cross_zero == 1:
            intercept_1 = 0
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
        
    elif breaking_point == 0: #where 0 means no
        x_name = df.columns[0]
        y_name = df.columns[1]
        x_values = list(df[x_name].values)
        y_values = list(df[y_name].values)
        slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = linregress(x_values, y_values)
        xfit_1 = np.linspace(x_values[0], x_values[-1], 100)
        cross_zero = eval(input('Do you need intercept to be zero? YES(1) or NO(0):'))
        if cross_zero == 1:
            intercept_1 = 0
        else:
            pass
        yfit_1 = slope_1*xfit_1 + intercept_1
        plt.plot(xfit_1, yfit_1, 'r-', label= 'Linear Fit_1')
        plt.errorbar(x_values, y_values, fmt='o',label='Data point_1', xerr = 0, yerr = std_err_1, capsize = 5)
        # we don't want error on x axis
        # to make vertical line longer, we need to set parameter 'capsize' with appropriate value
        # show the function of the line
        plt.text(x_values[-4], y_values[-4]*0.90, 'y = ' + str(round(slope_1,3)) + 'x + ' + str(round(intercept_1,3)) )
        
        # set unit of each axis
        Unit_x = input('Please input unit of X axis(e.g.[m]):')
        Unit_y = input('Please input unit of Y axis(e.g.[m]):')
        
        print('-----------------------------------\n')
        print('Slope_1, Error_1, Intercept_1, Correlation Coefficient_1:') 
        # r^2 = coefficient of determination
        print(slope_1, std_err_1, intercept_1, r_value_1**2)
        print('-----------------------------------\n')
        
        #calculate std_err_on_y of line 1
        def power_1(a):
            return a**slope_1 + intercept_1
        y_fit_values_1 = list(map(power_1,x_values)) #this is a list which only contain several fitted points
        y_y_fit_one = [y_values[i]-y_fit_values_1[i] for i in range(len(y_fit_values_1))] #Put 'y - y_fit' of line 1 into list
        summa_one = [i**2 for i in y_y_fit_one]
        total_one = 0
        for i in range(len(summa_one)):
            total_one += summa_one[i]
        std_err_on_y_1 = math.sqrt(total_one / (len(y_fit_values_1) - 2))
        
        
        #calculate chi-squared of line 1
        chi_squ_1 = 0
        for i in range(len(x_values)):
            chi_squ_1 += ((y_values[i] - y_fit_values_1[i])/std_err_on_y_1)**2
        print('chi square test: {}'.format(round(chi_squ_1,2)))# two digits
        
        plt.grid()
        plt.xlabel(x_name + Unit_x)
        plt.ylabel(y_name + Unit_y)
        plt.legend(loc = 'best')
        plt.show()
        
    else:
        print('Please make sure that your input is legal')
        
    
if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print('No such file found!')
    except ValueError as e:
        print('Values input should be less than range of index!')
    
    