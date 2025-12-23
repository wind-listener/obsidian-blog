"""
Author: Benjamin zzm_88orz@163.com
Date: 2024-08-18 21:47:30
LastEditors: Benjamin zzm_88orz@163.com
LastEditTime: 2024-08-18 21:54:20
FilePath: \编程基础四大件\数据结构和算法\算法\Leetcode\array\medium\718.最长重复子数组.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""
#
# @lc app=leetcode.cn id=718 lang=python3
#
# [718] 最长重复子数组
#
# https://leetcode.cn/problems/maximum-length-of-repeated-subarray/description/
#
# algorithms
# Medium (56.74%)
# Likes:    1103
# Dislikes: 0
# Total Accepted:    260.2K
# Total Submissions: 458.6K
# Testcase Example:  '[1,2,3,2,1]\n[3,2,1,4,7]'
#
# 给两个整数数组 nums1 和 nums2 ，返回 两个数组中 公共的 、长度最长的子数组的长度 。
# 
# 
# 
# 示例 1：
# 
# 
# 输入：nums1 = [1,2,3,2,1], nums2 = [3,2,1,4,7]
# 输出：3
# 解释：长度最长的公共子数组是 [3,2,1] 。
# 
# 
# 示例 2：
# 
# 
# 输入：nums1 = [0,0,0,0,0], nums2 = [0,0,0,0,0]
# 输出：5
# 
# 
# 
# 
# 提示：
# 
# 
# 1 <= nums1.length, nums2.length <= 1000
# 0 <= nums1[i], nums2[i] <= 100
# 
# 
#

# @lc code=start
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        m,n = len(nums1), len(nums2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        max_length = 0
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if nums1[i-1] ==nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1]+1
                    max_length = max(max_length, dp[i][j])

        return max_length
# @lc code=end

