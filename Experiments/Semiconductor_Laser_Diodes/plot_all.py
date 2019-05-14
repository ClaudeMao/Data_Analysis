import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import re

def main():
    current_path_of_py = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(current_path_of_py)
    a = input('input all file names:')
    # e.g: 5.xlsx,8.xlsx,11.xlsx,14.xlsx,17.xlsx,20.xlsx,23.xlsx,26.xlsx,29.xlsx,32.xlsx,35.xlsx
    filename = a.split(',')
    def pick_temp(x):
        re_temp = re.compile('^([0-9]+)\.xlsx$')
        return re_temp.match(x).group(1)
    for i in range(len(filename)):
        worksheet = pd.read_excel(filename[i])
        df = pd.DataFrame(worksheet)
        x_name = df.columns[0]
        y_name = df.columns[1]
        x_values = list(df[x_name].values)
        y_values = list(df[y_name].values)
        labelll = pick_temp(filename[i])
        plt.plot(x_values, y_values, lw = 1, label = str(labelll)+'℃')
    plt.legend(loc = 'best')
    plt.grid()
    plt.xlabel('Output Voltage[V]')
    plt.ylabel('Photodiode Voltage[V]')
    plt.legend(loc = 'best')
    plt.show()
    
if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print('No such file found!')
    except ValueError as e:
        print('Values input should be less than range of index!')
    
    