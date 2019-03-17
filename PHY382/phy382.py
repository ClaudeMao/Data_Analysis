import numpy as np
import matplotlib.pyplot as plt

Temp = input('please input temperature:')
TempStrList = [j for j in Temp.split()]
TempList = [eval(j) for j in Temp.split()]

V = np.linspace(-0.1, 0.5, 2000)
# I suggest start point is -0.1 and end point is 0.5 for temperature 300K and 400K
# because exponential increase rapidly and image will be bad if start and end is not appropriate
for i in range(len(TempList)):
    beta = 1.6*10**-19/(1.38*10**-23*TempList[i])
    I0 = 2/(np.exp(beta*0.5)-1)
    f = lambda x:I0*x*(np.exp(beta*x)-1)-2*x
    P = f(V)
    plt.plot(V, P, label= TempStrList[i]+'K')
    Pmax = min(P)
    Vmax_index = int(np.argwhere(P==Pmax))
    Vmax = V[Vmax_index]
    print(Vmax,Pmax)
    plt.plot(Vmax, Pmax, 'o', label = 'Pmax at '+TempStrList[i]+'K')
    plt.text(Vmax*0.99, Pmax*0.99, '(' + str(round(Vmax,5)) + ',' + str(round(Pmax,5)) + ')')
plt.grid()
plt.xlabel(r'Voltage[V]')
plt.ylabel(r'Power[W]')
plt.legend(loc = 'best')
plt.show()