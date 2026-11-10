"""Integration tests across modules."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from biokit.annotation import AnnotationFeature, translate_cds
from biokit.assembly import DeBruijnAssembler, assembly_gc, l50, n50
from biokit.crispr import design_guides, find_off_targets
from biokit.orf import ORFFinder
from biokit.primer import design_primer
from biokit.rna import transcribe
from biokit.translation import Translator
from biokit.utils.io_utils import make_record
from biokit.visualization import plot_gc_content


def teardown_function(_):
    plt.close("all")


def test_assembly_to_visualization():
    reads = [
        "ATGGCAGGTGACCCG",
        "GCAGGTGACCCGTGA",
        "GTGACCCGTGAATGA",
        "CCCGTGAATGAAACG",
        "TGAATGAAACGTACG",
    ]
    result = DeBruijnAssembler(k=5).assemble(reads)
    assert len(result.contigs) > 0
    assert n50(result.contigs) > 0
    assert l50(result.contigs) >= 1
    assert 0.0 <= assembly_gc(result.contigs) <= 1.0

    contig = result.contigs[0]
    fig = plot_gc_content(str(contig.seq), window=10)
    assert fig is not None


def test_rna_to_primer_workflow():
    genome = "ATGGCAGGTGACCCGTGAATGAAACGTACGTTAAGGATGCAA" * 2
    orfs = ORFFinder(minimum_length=6).find(genome)
    assert len(orfs) > 0
    assert orfs[0].protein_sequence

    primers = design_primer(genome, length=20, tm_target=55.0)
    assert len(primers) > 0

    mrna = transcribe(genome)
    assert "U" in mrna


def test_crispr_off_target_search():
    target = "ATGGCAGGTGACCCGTTGACCGGTAACGCATGCAGTGGACCTAGG"
    guides = design_guides(target)
    assert len(guides) > 0
    guide = guides[0]
    off_target_seq = "TT" + guide.spacer[2:] + guide.pam
    reference = target + off_target_seq
    offs = find_off_targets(guide, reference, max_mismatches=3)
    assert len(offs) > 0


def test_translation_consistency():
    cds = "ATGGCAGGTGACCCGTGA"
    assert translate_cds(cds) == Translator().translate(cds)


def test_assembly_metrics_consistent():
    lengths = [100, 200, 300, 400]
    records = [make_record("A" * length, str(i)) for i, length in enumerate(lengths)]
    assert n50(lengths) == n50(records)
    assert l50(lengths) == l50(records)


def test_annotation_feature_lookup():
    from biokit.annotation import feature_children, features_by_type, features_in_range

    feats = [
        AnnotationFeature(
            seqid="chr1", type="gene", start=1, end=100, strand="+", attributes={"ID": "g1"}
        ),
        AnnotationFeature(
            seqid="chr1",
            type="mRNA",
            start=1,
            end=100,
            strand="+",
            attributes={"ID": "m1", "Parent": "g1"},
        ),
    ]
    assert len(features_by_type(feats, "gene")) == 1
    assert len(features_in_range(feats, start=1, end=200, seqid="chr1")) == 2
    assert len(feature_children(feats, "g1")) == 1


import pytest


@pytest.mark.asyncio
async def test_tp53_golden_path_workflow():
    from nexus.workflows.analyze import AnalyzeBiomedicalEntityWorkflow

    workflow = AnalyzeBiomedicalEntityWorkflow()
    context = await workflow.run_context("TP53")

    assert context.query == "TP53"
    assert context.subject is not None
    assert context.graph is not None
    assert context.report is not None
    assert context.report.scientific_confidence is not None
    assert len(context.entities) > 0
