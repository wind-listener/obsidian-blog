#
# @lc app=leetcode.cn id=52 lang=python3
#
# [52] N 皇后 II
#
# https://leetcode.cn/problems/n-queens-ii/description/
#
# algorithms
# Hard (82.34%)
# Likes:    519
# Dislikes: 0
# Total Accepted:    155.2K
# Total Submissions: 188.5K
# Testcase Example:  '4'
#
# n 皇后问题 研究的是如何将 n 个皇后放置在 n × n 的棋盘上，并且使皇后彼此之间不能相互攻击。
# 
# 给你一个整数 n ，返回 n 皇后问题 不同的解决方案的数量。
# 
# 
# 
# 
# 
# 示例 1：
# 
# 
# 输入：n = 4
# 输出：2
# 解释：如上图所示，4 皇后问题存在两个不同的解法。
# 
# 
# 示例 2：
# 
# 
# 输入：n = 1
# 输出：1
# 
# 
# 
# 
# 提示：
# 
# 
# 1 <= n <= 9
# 
# 
# 
# 
#

# @lc code=start
class Solution:
    def totalNQueens(self, n: int) -> int:
        def solve(row: int, columns: int, diagonal_1: int, diagonal_2: int) -> int:
            if row == n:
                return 1
            else:
                count = 0
                availablePositions = ((1 << n) - 1) & (~(columns | diagonal_1 | diagonal_2))
                while availablePositions:
                    position = availablePositions & (-availablePositions)
                    availablePositions = availablePositions & (availablePositions - 1)
                    count += solve(row + 1, columns | position, (diagonal_1 | position) << 1, (diagonal_2 | position) >> 1)
                return count
        
        return solve(0, 0, 0, 0)
# @lc code=end

