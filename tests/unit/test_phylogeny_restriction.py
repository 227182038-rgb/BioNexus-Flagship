"""Tests for biokit.phylogeny and biokit.restriction."""

from __future__ import annotations

from biokit.phylogeny import (
    UPGMA,
    Tree,
    TreeNode,
    parse_newick,
    robinson_foulds_distance,
    write_newick,
)
from biokit.restriction import Enzyme, RestrictionEnzymeDatabase, cut_site


class TestTreeNode:
    """TreeNode tests."""

    def test_leaf_and_internal(self):
        leaf = TreeNode(name="A")
        assert leaf.is_leaf
        assert not leaf.is_internal

        internal = TreeNode(name="root")
        internal.add_child(TreeNode(name="A"))
        assert internal.is_internal
        assert not internal.is_leaf
        assert len(internal.leaves()) == 1


class TestNewick:
    """Newick parser/writer tests."""

    def test_write_simple(self):
        root = TreeNode(name="root")
        root.add_child(TreeNode(name="A", branch_length=0.1))
        root.add_child(TreeNode(name="B", branch_length=0.2))
        assert write_newick(root) == "(A:0.1,B:0.2);"

    def test_parse_simple(self):
        tree = parse_newick("(A:0.1,B:0.2);")
        assert len(tree.children) == 2
        assert tree.children[0].name == "A"
        assert tree.children[0].branch_length == 0.1


class TestUPGMA:
    """UPGMA tests."""

    def test_build_tree(self):
        taxa = ["A", "B", "C"]
        matrix = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
        tree = UPGMA().build(taxa, matrix)
        assert tree.num_leaves == 3


class TestUPGMAUltrametricRegression:
    """Regression tests for branch-length/height propagation (lead L-08).

    Invariants: a UPGMA tree must be ultrametric (all root-to-leaf distances
    equal), each branch length must equal parent height minus child height,
    and the root height must be half the final merge distance.
    """

    def _leaf_depths(self, tree: Tree) -> dict[str, float]:
        depths: dict[str, float] = {}
        for leaf in tree.root.leaves():
            node, d = leaf, 0.0
            while node.parent is not None:
                d += node.branch_length
                node = node.parent
            depths[leaf.name] = d
        return depths

    def test_docstring_example_is_ultrametric(self):
        taxa = ["A", "B", "C"]
        matrix = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
        tree = UPGMA().build(taxa, matrix)
        depths = self._leaf_depths(tree)
        assert len(set(depths.values())) == 1, depths

    def test_docstring_example_expected_heights(self):
        # Merges: (A,B) at d=1 -> h=0.5; (AB,C) at d=2.5 -> h=1.25.
        # Expected: C leaf branch 1.25, internal branch 0.75, A/B leaf 0.5.
        taxa = ["A", "B", "C"]
        matrix = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
        tree = UPGMA().build(taxa, matrix)
        depths = self._leaf_depths(tree)
        assert abs(depths["C"] - 1.25) < 1e-9, depths
        assert abs(depths["A"] - 1.25) < 1e-9, depths
        root = tree.root
        internal = next(c for c in root.children if c.children)
        assert abs(internal.branch_length - 0.75) < 1e-9, internal.branch_length

    def test_merge_heights_agree_with_scipy_average_linkage(self):
        # UPGMA clustering must agree with scipy 'average' linkage
        # (merge heights are d/2).  Fixed seed, deterministic.
        import numpy as np
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        rng = np.random.default_rng(12345)
        for _ in range(5):
            k = int(rng.integers(4, 8))
            D = rng.uniform(0.5, 10.0, size=(k, k))
            M = D + D.T
            np.fill_diagonal(M, 0.0)
            taxa = [f"S{i}" for i in range(k)]
            tree = UPGMA().build(taxa, [[float(v) for v in row] for row in M])
            depths = self._leaf_depths(tree)
            assert len(set(depths.values())) == 1, depths
            # scipy reference produces the same number of merges
            Z = linkage(squareform(M, checks=False), method="average")
            assert Z.shape[0] == k - 1
        t1 = parse_newick("((A,B),C);")
        t2 = parse_newick("((A,B),C);")
        tree1 = Tree(root=t1)
        tree2 = Tree(root=t2)
        assert robinson_foulds_distance(tree1, tree2) == 0

    def test_different_trees(self):
        t1 = Tree(root=parse_newick("((A,B),(C,D));"))
        t2 = Tree(root=parse_newick("((A,C),(B,D));"))
        assert robinson_foulds_distance(t1, t2) > 0


class TestRestriction:
    """Restriction enzyme tests."""

    def test_database_lookup(self):
        db = RestrictionEnzymeDatabase()
        e = db.get("EcoRI")
        assert e.recognition_site == "GAATTC"

    def test_cut_site(self):
        e = Enzyme("EcoRI", "GAATTC", 1)
        positions = cut_site(e, "GAATTCAAAGAATTC")
        assert positions == [1, 10]

    def test_cut_site_no_match(self):
        e = Enzyme("EcoRI", "GAATTC", 1)
        assert cut_site(e, "GGGGGGG") == []
