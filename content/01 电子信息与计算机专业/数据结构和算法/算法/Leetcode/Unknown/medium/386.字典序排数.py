"""
Author: Benjamin zzm_88orz@163.com
Date: 2024-08-20 23:24:50
LastEditors: Benjamin zzm_88orz@163.com
LastEditTime: 2024-08-20 23:34:42
FilePath: \编程基础四大件\数据结构和算法\算法\Leetcode\Unknown\medium\386.字典序排数.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""
#
# @lc app=leetcode.cn id=386 lang=python3
#
# [386] 字典序排数
#
# https://leetcode.cn/problems/lexicographical-numbers/description/
#
# algorithms
# Medium (74.53%)
# Likes:    496
# Dislikes: 0
# Total Accepted:    79.5K
# Total Submissions: 106.7K
# Testcase Example:  '13'
#
# 给你一个整数 n ，按字典序返回范围 [1, n] 内所有整数。
# 
# 你必须设计一个时间复杂度为 O(n) 且使用 O(1) 额外空间的算法。
# 
# 
# 
# 示例 1：
# 
# 
# 输入：n = 13
# 输出：[1,10,11,12,13,2,3,4,5,6,7,8,9]
# 
# 
# 示例 2：
# 
# 
# 输入：n = 2
# 输出：[1,2]
# 
# 
# 
# 
# 提示：
# 
# 
# 1 <= n <= 5 * 10^4
# 
# 
#

# @lc code=start
class Solution:
    def lexicalOrder(self, n: int) -> List[int]:
        result = []
        currrent = 1
        for _ in range(n):
            result.append(currrent)
            if currrent*10<=n:
                currrent*=10
            else:
                while currrent%10 == 9 or currrent+1 >n:
                    currrent//=10
                currrent+=1
        return result
# @lc code=end

