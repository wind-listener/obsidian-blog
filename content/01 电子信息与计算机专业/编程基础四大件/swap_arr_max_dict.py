def solve(n,arr):
    for i, num in enumerate(arr):
        if i+1<n and arr[i]<arr[i+1]:
            arr[i],arr[i+1] = arr[i+1], arr[i]
            i+
        

n = int(input())
arr = [int(char) for char in input().split()]

