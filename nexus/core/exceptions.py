"""BioNexus exception hierarchy."""

from __future__ import annotations


class BioNexusError(Exception):
    """Base exception for all BioNexus errors."""


# ---------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------

class ConfigurationError(BioNexusError):
    """Invalid or missing configuration."""


class ValidationError(BioNexusError):
    """Input validation failed."""


class RegistryError(BioNexusError):
    """Registry operation failed."""


class CacheError(BioNexusError):
    """Cache operation failed."""


# ---------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------

class KnowledgeError(BioNexusError):
    """Knowledge retrieval failed."""


class PubMedError(KnowledgeError):
    """PubMed request failed."""


class UniProtError(KnowledgeError):
    """UniProt request failed."""


class PDBError(KnowledgeError):
    """Protein Data Bank request failed."""


class EnsemblError(KnowledgeError):
    """Ensembl request failed."""


class GOError(KnowledgeError):
    """Gene Ontology request failed."""


class NCBIError(KnowledgeError):
    """NCBI request failed."""


# ---------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------

class FusionError(BioNexusError):
    """Entity fusion failed."""


# ---------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------

class GraphError(BioNexusError):
    """Knowledge graph operation failed."""


class NodeNotFoundError(GraphError):
    """Requested node does not exist."""


class EdgeError(GraphError):
    """Graph edge operation failed."""


# ---------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------

class ReasoningError(BioNexusError):
    """Scientific reasoning failed."""


class EvidenceError(ReasoningError):
    """Evidence processing failed."""


class HypothesisError(ReasoningError):
    """Hypothesis generation failed."""


# ---------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------

class WorkflowError(BioNexusError):
    """Workflow execution failed."""


# ---------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------

class PluginError(BioNexusError):
    """Plugin operation failed."""


class PluginRegistrationError(PluginError):
    """Plugin registration failed."""


class PluginExecutionError(PluginError):
    """Plugin execution failed."""
