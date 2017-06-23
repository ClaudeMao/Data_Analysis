from turtle import *
setup(650,350,200,200)
up()
fd(-250)
down()
width(10)
pencolor('red')# 'colorstring'或者（r,g,b）red green blue
seth(-40)
for i in range(4):
    pencolor('red')
    circle(40,80)
    pencolor('yellow')
    circle(-40,80)	
circle(40,80/2)
fd(40)
pencolor('blue')
circle(16,180)
fd(40*2/3)
