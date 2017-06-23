from random import *
print('猜数字')
guesstime=1
seed(1)
prex=randint(0,100)
try:
    while True:
        guess=eval(input('输入一个0-100的数字：'))
        if guess>prex:
            print('大了,继续猜')
            guesstime+=1
        elif guess<prex:
             print('小了,继续猜')
             guesstime+=1
        else:
            print('狗东西居然被你才对了')
            break
    print('你一共猜了{}次'.format(guesstime))
except NameError:
    print('你个zz输的什么东西')
    
