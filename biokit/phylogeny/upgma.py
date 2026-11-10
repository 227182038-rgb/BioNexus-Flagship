"""UPGMA tree builder."""

from __future__ import annotations

from biokit.phylogeny.node import TreeNode
from biokit.phylogeny.tree import Tree


class UPGMA:
    """Unweighted Pair Group Method with Arithmetic Mean.

    Builds a rooted tree from a distance matrix.

    Examples
    --------
    >>> taxa = ["A", "B", "C"]
    >>> matrix = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
    >>> tree = UPGMA().build(taxa, matrix)
    >>> tree.num_leaves
    3
    """

    def build(self, taxa: list[str], matrix: list[list[float]]) -> Tree:
        """Build a UPGMA tree from a distance matrix.

        Heights: a merge at distance ``d`` creates a node of height ``d/2``;
        each child's branch length equals the new height minus the child's
        *accumulated* height (0 for leaves), which keeps the tree ultrametric.
        """
        if len(taxa) != len(matrix) or any(len(row) != len(taxa) for row in matrix):
            raise ValueError("matrix dimensions must match taxa")
        nodes: list[TreeNode] = [TreeNode(name=t) for t in taxa]
        # Height (distance from the tips) of each cluster's representative.
        # Leaves have height 0; a merge at distance d produces height d/2.
        heights: list[float] = [0.0] * len(nodes)
        sizes: list[int] = [1] * len(taxa)
        dist = [row[:] for row in matrix]
        while len(nodes) > 1:
            # Find closest pair (ties resolved to the first minimum; a
            # deterministic but arbitrary choice among equally optimal merges).
            min_i, min_j, min_d = 0, 1, float("inf")
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if dist[i][j] < min_d:
                        min_d = dist[i][j]
                        min_i, min_j = i, j
            # Create new node at height d/2
            new_node = TreeNode(branch_length=0.0)
            new_height = min_d / 2.0
            child_i = nodes[min_i]
            child_j = nodes[min_j]
            # Branch length = new height - child's accumulated height so that
            # every root-to-leaf path sums to the same total (ultrametricity).
            child_i.branch_length = new_height - heights[min_i]
            child_j.branch_length = new_height - heights[min_j]
            new_node.add_child(child_i)
            new_node.add_child(child_j)
            # Update distances (weighted arithmetic mean by cluster size)
            new_size = sizes[min_i] + sizes[min_j]
            new_dist: list[float] = []
            for k in range(len(nodes)):
                if k in (min_i, min_j):
                    continue
                d = (sizes[min_i] * dist[min_i][k] + sizes[min_j] * dist[min_j][k]) / new_size
                new_dist.append(d)
            # Remove old nodes, add new
            keep = [k for k in range(len(nodes)) if k != min_i and k != min_j]
            nodes = [nodes[k] for k in keep] + [new_node]
            sizes = [sizes[k] for k in keep] + [new_size]
            heights = [heights[k] for k in keep] + [new_height]
            new_matrix = [[dist[a][b] for b in keep] for a in keep]
            new_matrix.append([*new_dist, 0.0])
            for idx, d in enumerate(new_dist):
                new_matrix[idx].append(d)
            dist = new_matrix
        return Tree(root=nodes[0])


__all__ = ["UPGMA"]
