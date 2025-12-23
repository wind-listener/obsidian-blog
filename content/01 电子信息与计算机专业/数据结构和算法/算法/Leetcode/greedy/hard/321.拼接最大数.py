"""
Author: Benjamin zzm_88orz@163.com
Date: 2024-08-11 18:19:41
LastEditors: Benjamin zzm_88orz@163.com
LastEditTime: 2024-08-11 18:20:39
FilePath: \编程基础四大件\数据结构和算法\算法\Leetcode\greedy\hard\321.拼接最大数.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""
#
# @lc app=leetcode.cn id=321 lang=python3
#
# [321] 拼接最大数
#
# https://leetcode.cn/problems/create-maximum-number/description/
#
# algorithms
# Hard (42.23%)
# Likes:    589
# Dislikes: 0
# Total Accepted:    42.2K
# Total Submissions: 99.9K
# Testcase Example:  '[3,4,6,5]\n[9,1,2,5,8,3]\n5'
#
# 给你两个整数数组 nums1 和 nums2，它们的长度分别为 m 和 n。数组 nums1 和 nums2
# 分别代表两个数各位上的数字。同时你也会得到一个整数 k。
# 
# 请你利用这两个数组中的数字中创建一个长度为 k <= m + n 的最大数，在这个必须保留来自同一数组的数字的相对顺序。
# 
# 返回代表答案的长度为 k 的数组。
# 
# 
# 
# 示例 1：
# 
# 
# 输入：nums1 = [3,4,6,5], nums2 = [9,1,2,5,8,3], k = 5
# 输出：[9,8,6,5,3]
# 
# 
# 示例 2：
# 
# 
# 输入：nums1 = [6,7], nums2 = [6,0,4], k = 5
# 输出：[6,7,6,0,4]
# 
# 
# 示例 3：
# 
# 
# 输入：nums1 = [3,9], nums2 = [8,9], k = 3
# 输出：[9,8,9]
# 
# 
# 
# 
# 提示：
# 
# 
# m == nums1.length
# n == nums2.length
# 1 <= m, n <= 500
# 0 <= nums1[i], nums2[i] <= 9
# 1 <= k <= m + n
# 
# 
#

# @lc code=start
class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        # 找到 nums 数组中长度为 t 的最大子序列
        def maxSubsequence(nums, t):
            stack = []
            drop = len(nums) - t
            for num in nums:
                while drop > 0 and stack and stack[-1] < num:
                    stack.pop()
                    drop -= 1
                stack.append(num)
            return stack[:t]
        
        # 合并两个子序列，使得结果的字典序最大
        def merge(subseq1, subseq2):
            return [max(subseq1, subseq2).pop(0) for _ in range(len(subseq1) + len(subseq2))]
        
        # 尝试不同的拆分方式，找出最大结果
        max_sequence = []
        for i in range(max(0, k - len(nums2)), min(k, len(nums1)) + 1):
            subseq1 = maxSubsequence(nums1, i)
            subseq2 = maxSubsequence(nums2, k - i)
            max_sequence = max(max_sequence, merge(subseq1, subseq2))
        
        return max_sequence

# @lc code=end

