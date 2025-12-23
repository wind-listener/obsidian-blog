from collections import defaultdict

n,m,k = (int(char) for char in input().split())
apples = defaultdict(list)

for _ in range(m):
    position, heights = [int(char) for char in input().split()]
    apples[position].append(heights)

total_count=0
for postion, heights in apples.items():
    heights.sort()
    max_count = 0
    for i, start_height in enumerate(heights):
        now_count = 0
        while heights[i]<=start_height+k:
            i+=1
            now_count+=1
        max_count = max(now_count, max_count)
    total_count += max_count

print(total_count)