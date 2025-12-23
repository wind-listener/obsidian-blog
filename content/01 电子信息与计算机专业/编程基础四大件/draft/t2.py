n, k = (int(char) for char in input().split())
values = [int(char) for char in input().split()]

dp = [float('inf')]*(n+1)
dp[0] = 0

from collections import deque
max_queue = deque()
min_queue = deque()

for i in range(1,n+1):
    max_queue.clear()
    min_queue.clear()

    for j in range(i,0,-1):
        while max_queue and values[max_queue[-1]-1]<=values[j-1]:
            max_queue.pop()
        max_queue.append(j)
        while min_queue and values[min_queue[-1]-1]>=values[j-1]:
            min_queue.pop()
        min_queue.append(j)

        if values[max_queue[0]-1] - values[min_queue[0]-1] > k:
            break
        
        dp[i] = min(dp[i],dp[j-1]+1)

print(dp[n])