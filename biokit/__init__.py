"""BioKit 2.0 — a market-competitive Python toolkit for biology.

Public modules:

* :mod:`biokit.sequence` — :class:`BioSequence` value type and operations
* :mod:`biokit.io` — file format parsers (FASTA, FASTQ, GFF3, BED, GenBank, ...)
* :mod:`biokit.alignment` — Needleman-Wunsch, Smith-Waterman, Gotoh, MSA
* :mod:`biokit.assembly` — De Bruijn graph, OLC assemblers, N50/L50
* :mod:`biokit.annotation` — feature orchestration across formats
* :mod:`biokit.orf` — six-frame ORF finding
* :mod:`biokit.translation` — codon-based translation
* :mod:`biokit.codon` — codon usage analysis (RSCU, CAI)
* :mod:`biokit.kmers` — k-mer counting and indexing
* :mod:`biokit.motifs` — motif finding and PSSM
* :mod:`biokit.blast` — BLAST XML/tabular parsing, local runner
* :mod:`biokit.primer` — primer design with Tm (Wallace/GC/NN)
* :mod:`biokit.rna` — transcription, reverse transcription, structure
* :mod:`biokit.structural` — PDB parsing, RMSD, Kabsch superposition
* :mod:`biokit.popgen` — HWE, F_ST, linkage disequilibrium
* :mod:`biokit.crispr` — sgRNA design, off-target search
* :mod:`biokit.machine_learning` — classifiers, clustering, encoders
* :mod:`biokit.phylogeny` — UPGMA, Robinson-Foulds (neighbor-joining is not implemented)
* :mod:`biokit.restriction` — restriction enzyme analysis
* :mod:`biokit.visualization` — matplotlib plots (GenomeDiagram is not implemented)
* :mod:`biokit.statistics` — sequence composition, distributions

For the command-line interface run ``biokit --help``.
"""

from __future__ import annotations

from biokit.exceptions import (
    AnnotationError,
    AssemblyError,
    BioKitError,
    BlastError,
    ConvergenceError,
    CRISPRError,
    EncodingError,
    InvalidFormatError,
    InvalidSequenceError,
    NotFittedError,
    PopGenError,
    PrimerError,
    RNAError,
    StructuralError,
    ValidationError,
    VisualizationError,
)
from biokit.sequence import DNA, RNA, BioSequence, Protein
from biokit.types import PathLike

__version__ = "2.0.0"
__all__ = [
    "__version__",
    # Exceptions
    "BioKitError",
    "InvalidFormatError",
    "InvalidSequenceError",
    "AnnotationError",
    "AssemblyError",
    "BlastError",
    "PrimerError",
    "RNAError",
    "StructuralError",
    "PopGenError",
    "CRISPRError",
    "VisualizationError",
    "ValidationError",
    "NotFittedError",
    "ConvergenceError",
    "EncodingError",
    # Core types
    "BioSequence",
    "DNA",
    "RNA",
    "Protein",
    "PathLike",
]
