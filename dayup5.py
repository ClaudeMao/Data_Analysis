def dayUp(df):
    dayup=0.01
    for i in range(365):
        if i%7 in [1,2,3,4,5]:
            dayup=dayup*(1+df)
        else:
            dayup=dayup*(1-0.01)
    return dayup
dayfactor=0.01
while(dayUp(dayfactor)<37.78):
    dayfactor+=0.0001
print('每天的努力参数是：{:.4f}'.format(dayfactor))
