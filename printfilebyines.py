fname = input("请输入要打开的文件: ")
fo = open(fname, "r")
for line in fo:
    print(line)
fo.close()
