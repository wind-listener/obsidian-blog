import math
import itertools

def count_access_num(n:int, x:int, scores:list) -> int:
    # # scores.sort()
    # # scores = scores[::-1]
    # scores = sorted(scores, reverse=True)
    # plan = scores[:math.ceil(n/2)]
    # if sum(plan)<x:
    #     return 0
    # result = 0
   
    # def dfs(plan):


    # for i in range()
    count = 0   
    for plan in list(itertools.combinations(scores)):
        if sum(plan)>=x:
            count+=1
    return count


T = int(input())
for _ in range(T):
    n, x = [int(char) for char in input().split()]
    scores = [int(char) for char in input().split()]
    print(count_access_num(n, x, scores))
    