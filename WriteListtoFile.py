fname=input('请输入要写入的文件：')
fo=open(fname,'r+')
ls=['有唐诗宋词元曲']
fo.writelines(ls)
fo.seek(0)
for line in fo:
    print(line)
fo.close()
