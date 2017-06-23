from turtle import *
setup(0.4,0.8,900,0)
seth(90)
x=200
for i in range(20):
    fd(x-10*i)
    right(90)
    fd(x-10*i)
    right(90)
    speed(0.01)

