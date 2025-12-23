# 4
# 1 2 3 4
# 1 2
# 2 3
# 2 4

import sys
from collections import defaultdict, Counter
class TreeNode:
    def __init__(self, color=0) -> None:
        self.color = color
        self.children = []

def build_tree(n, colors, edges):
    tree = defaultdict(TreeNode)
    for i in range(1, n + 1):
        tree[i] = TreeNode(color=colors[i - 1])
    for a, b in edges:
        tree[a].children.append(tree[b])
        tree[b].children.append(tree[a])

    return tree


def dfs(node, parent, tree, result):
    color_count = Counter()
    xor_sum = 0

    for child in tree[node].children:
        if child != parent:
            subtree_colors, subtree_xor = dfs(child, node, tree, result)
            color_count.update(subtree_colors)

    color_count[tree[node].color] += 1
    xor_sum ^= tree[node].color

    if color_count:
        most_common_color, count = color_count.most_common(1)[0]
        removed_xor_sum = xor_sum
        removed_xor_sum ^= most_common_color * count
        result[0] = max(result[0], removed_xor_sum)

    return color_count, xor_sum


n = int(input())
colors = [int(char) for char in input().split()]
edges = []
for line in sys.stdin:
    edges.append([int(char) for char in line.split()])

# for _ in range(n):
#     edges.append([int(char) for char in input().split()])

tree = build_tree(n, colors=colors, edges=edges)
result = [0]
dfs(1, None, tree, result)
print(result[0])
