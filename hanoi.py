def hanoi(n,a,b,c):
    if n==1:
        print(a,'-->',c)
    else:
        hanoi(n-1,a,c,b)
        hanoi(1,a,b,c)
        hanoi(n-1,b,a,c)
n=eval(input('输入A塔上的初始层数n：'))
from time import clock
clock()
hanoi(n,'a','b','c')
print('运算时间为：{}'.format(clock()))




