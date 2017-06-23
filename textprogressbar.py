from time import *
scale=50
print('执行开始'.center(scale//2,'-'))
t=clock()
for i in range (scale+1):#i从0开始循环，若不加1，循环10次只到90%
    a,b='*'*i,'.'*(scale-i)         #接上，所以要加1，循环11次才能输出100%
    c=(i/scale)*100
    t-=clock()
    print('\r[{1}->{2}]{0:^3.0f}% {3:.2f}s'.format(c,a,b,-t),end='')
    sleep(0.05)
print('\n'+'执行结束'.center(scale//2,'-'))
