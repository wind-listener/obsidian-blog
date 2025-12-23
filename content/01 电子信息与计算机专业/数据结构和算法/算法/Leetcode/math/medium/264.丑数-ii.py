"""
Author: Benjamin zzm_88orz@163.com
Date: 2024-08-13 23:15:39
LastEditors: Benjamin zzm_88orz@163.com
LastEditTime: 2024-08-13 23:30:44
FilePath: \编程基础四大件\数据结构和算法\算法\Leetcode\math\medium\264.丑数-ii.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""
#
# @lc app=leetcode.cn id=264 lang=python3
#
# [264] 丑数 II
#
# https://leetcode.cn/problems/ugly-number-ii/description/
#
# algorithms
# Medium (58.18%)
# Likes:    1200
# Dislikes: 0
# Total Accepted:    183.8K
# Total Submissions: 315.9K
# Testcase Example:  '10'
#
# 给你一个整数 n ，请你找出并返回第 n 个 丑数 。
# 
# 丑数 就是质因子只包含 2、3 和 5 的正整数。
# 
# 
# 
# 示例 1：
# 
# 
# 输入：n = 10
# 输出：12
# 解释：[1, 2, 3, 4, 5, 6, 8, 9, 10, 12] 是由前 10 个丑数组成的序列。
# 
# 
# 示例 2：
# 
# 
# 输入：n = 1
# 输出：1
# 解释：1 通常被视为丑数。
# 
# 
# 
# 
# 提示：
# 
# 
# 1 <= n <= 1690
# 
# 
#

# @lc code=start
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        # 动态规划，比较有意思，但还没有完全理解其正确性 HAZE
        dp = [0]*n
        dp[0] = 1
        i2 = i3 = i5 = 0
        
        for i in range(1,n):
            next_ugly = min(dp[i2]*2,dp[i3]*3,dp[i5]*5)
            dp[i] = next_ugly
            
            if next_ugly == dp[i2]*2:
                i2 +=1
            if next_ugly == dp[i3]*3:
                i3 +=1
            if next_ugly == dp[i5]*5:
                i5 +=1
        return dp[-1]
# @lc code=end

