# BioNexus Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-358%20passing-brightgreen.svg)](#)
[![Coverage](https://img.shields.io/badge/coverage-75%25-brightgreen.svg)](#)

**BioNexus** is a unified platform that pairs **BioKit 2.0** (deterministic
bioinformatics) with **NEXUS Ω** (AI-native biological interpretation
intelligence). **BioKit computes. Nexus understands.**

## Architecture

```
                    ┌──────────────────────────────────┐
                    │   User (SDK / CLI / API / Web)   │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │       NEXUS Orchestrator         │
                    │  (literature, validation,        │
                    │   experiment, workflow, report)  │
                    └──────┬───────────────────┬───────┘
                           │                   │
              ┌────────────▼──┐       ┌────────▼────────┐
              │  Intelligence │       │   RAG + Memory  │
              │  (interpret,  │       │  (PubMed, NCBI, │
              │   reason,     │       │   UniProt, PDB) │
              │   hypothesize)│       └─────────────────┘
              └────┬──────────┘
                   │
              ┌────▼──────────────────────────────────┐
              │            BRIDGE LAYER               │  ◀── integration
              │  (22+ BioKitProgram adapters)         │
              └────┬──────────────────────────────────┘
                   │
              ┌────▼──────────────────────────────────┐
              │            BioKit 2.0                 │  ◀── deterministic
              │  30 modules across 22 subpackages:    │      substrate
              │  sequence, alignment, assembly, BLAST,│
              │  primer, CRISPR, popgen, structural,  │
              │  phylogeny, machine_learning, ...    │
              └───────────────────────────────────────┘
```

The bridge layer is the *only* place where NEXUS learns about BioKit's
concrete API. It enforces the Prime Directive: BioKit computations stay
deterministic and content-addressed; NEXUS agents consume their outputs
as evidence but may not modify them.

## Installation

```bash
git clone https://github.com/227182038-rgb/BioNexus.git
cd BioNexus
pip install -e ".[dev,openai,anthropic,glm,ollama]"
```

## Quick Start

```python
from bridge import quickstart

# One call wires NEXUS + BioKit 2.0 together with all 22+ programs registered
nx = quickstart()

# Run a deterministic BioKit calculation
output = nx.run_biokit("gc_content", {"sequence": "ATGGCAGGTGACCCGTGA"})
print(f"GC: {output.outputs['gc_percentage']:.2f}%")

# Find ORFs
orfs = nx.run_biokit("find_orfs", {
    "sequence": "ATGGCAGGTGACCCGTGAATGAAACGTACGTTGA",
    "minimum_length": 6,
})
print(f"Found {orfs.outputs['count']} ORFs")

# Ask NEXUS to interpret the result (literature-grounded, evidence-cited)
# (Uses DummyProvider by default — swap in OpenAI/Anthropic/GLM/Ollama for real LLM)
result = nx.interpret(
    "What does this GC content suggest about the sequence?",
    agent="literature",
    context={"biokit_output": output.outputs},
)
print(result.answer)
```

## CLI

```bash
# BioKit 2.0 commands
bionexus biokit gc --sequence ATGGCAGGT
bionexus biokit orfs --sequence ATGGCAGGTGACCCGTGA --min-length 6
bionexus biokit translate --sequence ATGGCAGGTGACCCGTGA

# NEXUS commands
bionexus nexus interpret --question "What does BRCA1 do?"
bionexus nexus agents

# Bridge commands
bionexus programs                                          # list all 23 registered BioKit programs
bionexus run gc_content --inputs '{"sequence": "ATGGCAGGT"}'
bionexus run find_orfs --inputs '{"sequence": "ATGGCAGGTGACCCGTGA", "minimum_length": 6}'
bionexus version
```

## Modules

### BioKit 2.0 (deterministic substrate)

23 subpackages: `sequence`, `io`, `alignment`, `assembly`, `annotation`,
`orf`, `translation`, `codon`, `kmers`, `motifs`, `blast`, `primer`, `rna`,
`structural`, `popgen`, `crispr`, `machine_learning`, `phylogeny`,
`restriction`, `visualization`, `statistics`, `models`, `utils`.

### NEXUS (AI interpretation layer)

10 subpackages: `core` (engine, orchestrator, BioKit facade),
`intelligence` (interpretation, validation, reasoning, hypothesis),
`rag` (indexing, retrieval, ranking, citation), `knowledge` (PubMed, NCBI,
UniProt, PDB, Ensembl, GO), `providers` (OpenAI, Anthropic, Gemini, GLM,
Ollama, OpenRouter), `agents` (literature, validation, experiment,
workflow, report), `memory`, `reports`, `plugins`, `api`.

### Bridge (integration layer)

22+ `BioKitProgram` adapters that wrap BioKit 2.0 modules as NEXUS
deterministic computation programs. Adding a new BioKit module to NEXUS
= adding one class in `bridge/biokit_programs.py`.

## License

Apache License 2.0. See [LICENSE](LICENSE).
