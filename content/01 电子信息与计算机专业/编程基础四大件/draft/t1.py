def min_cost_to_build_string(s, p, q):
    n = len(s)
    dp = [[float('inf')] * n for _ in range(n)]
    
    # 初始状态
    for i in range(n):
        dp[i][i] = p
    
    # 动态规划
    for length in range(2, n+1):  # 子串的长度
        for i in range(n-length+1):
            j = i + length - 1
            dp[i][j] = min(dp[i+1][j] + p, dp[i][j-1] + p)
            
            for k in range(1, length):
                if s[i:i+k] == s[j-k+1:j+1]:
                    dp[i][j] = min(dp[i][j], dp[i+k][j] + q, dp[i][j-k] + q)
    
    return dp[0][n-1]

# 示例

s = "bbcabc"
p = 3
q = 1
print(min_cost_to_build_string(s, p, q))  # 输出最小花费