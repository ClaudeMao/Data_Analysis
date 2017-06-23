try:
    num=eval(input('请输入一个数：'))
    print(num**2)
except NameError:
    print('输入错误，请输入一个整数！')
