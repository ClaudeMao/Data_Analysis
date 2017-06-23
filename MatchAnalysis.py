from random import random
def main():
    printintro()
    probA,probB,n=getinputs()
    winsA,winsB=simNgames(probA,probB,n)
    printsummary(winsA,winsB)
def printintro():
    print('这个程序模拟两个选手A和B的某种竞技比赛')
    print('程序运行需要A和B的能力值（以0到1之间小数表示）')
def getinputs():
    a=eval(input('请输入选手A的能力值：'))
    b=eval(input('请输入选手B的能力值：'))
    n=eval(input('请输入模拟比赛场次：'))
    return a,b,n
def simNgames(probA,probB,n):
    winsA,winsB=0,0
    for i in range(n):
        scoreA,scoreB=simOnegame(probA,probB)
        if scoreA>scoreB:
            winsA+=1
        else:
            winsB+=1
    return winsA,winsB
def simOnegame(probA,probB):
    scoreA,scoreB=0,0
    serving='A'
    while not gameover(scoreA,scoreB):
        if serving=='A':
            if random()<probA:
                scoreA+=1
            else:
                serving='B'
        else:
            if random()<probB:
                scoreB+=1
            else:
                serving='A'
    return scoreA,scoreB
def gameover(a,b):
    if a==15 or b==15:
        return a==15 or b==15
def printsummary(winsA,winsB):
    n=winsA+winsB
    print('竞技分析开始，共模拟{}场比赛'.format(n))
    print('A选手获胜{}场比赛，占比{:0.1%}'.format(winsA,winsA/n))
    print('B选手获胜{}场比赛，占比{:0.1%}'.format(winsB,winsB/n))
main()
    
