```python
  
# 输入为: 1 2 3 4 5
a = input()
# a = '1 2 3 4 5'

  
# 输入为： 1 2 3 4 5
a = input().split() # split()默认以空字符为分隔符，包括空格、换行(\n)、制表符(\t)等
# a = ['1', '2', '3', '4', '5']

# 输入为：1,2,3,4,5
b = input().split(',') # 以逗号为分隔符
# b = ['1', '2', '3', '4', '5']

# 输入为： 1 
a = int(input()) # 单个转换
	  
# 输入为：1 2 3 4 5
b = input().split() # b = ['1', '2', '3', '4', '5']
c = [int(i) for i in b] # 使用列表进行批量转换 c = [1, 2, 3, 4, 5]
d = [int(i) for i in input().split()] # 当然可以一步倒位
	  
# 使用map进行并行转换
e = map(int, input().split()) # 此时e是一个map迭代器，不能赋值，也不能索引
f = list(e) # 转换为列表，e = [1, 2, 3, 4, 5]
g = list(map(int, input().split())) # 一步到位

```

