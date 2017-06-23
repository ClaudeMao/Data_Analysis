from math import *
dayfactor=input('输入每天比前一天提高多少：')
dayup=pow(1+eval(dayfactor),365)
daydown=pow(1-eval(dayfactor),365)
print('dayup={:.3f},daydown={:.3f}'.format(dayup,daydown))
print('差距：{:.3f}'.format(dayup-daydown))
