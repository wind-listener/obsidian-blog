n = int(input())
cars = []
for _ in range(n):
    cars.append([int(char) for char in input().split()])

cars.sort(key=lambda x:x[0])
speeds = [car[1] for car in cars]

dp = [1]*n
for i in range(1, n):
    for j in range(i):
        if speeds[i]>speeds[j]:
            dp[i] = max(dp[i],dp[j]+1)
max_up_length = max(dp)
count = n - max_up_length
print(count)