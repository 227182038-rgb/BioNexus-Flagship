"""FastAPI application for the Nexus REST API.

Endpoints:

- ``GET /`` — web UI (HTML)
- ``GET /api/health`` — health check
- ``GET /api/agents`` — list registered agents
- ``POST /api/interpret`` — ask Nexus a question
- ``POST /api/validate`` — validate a claim
- ``POST /api/index`` — index a document
- ``GET /api/search?q=...`` — search the RAG index
- ``POST /api/biokit`` — run a BioKit program
- ``POST /api/report`` — generate a report
"""

from __future__ import annotations

from typing import Any
import logging
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from nexus.providers.dummy import DummyProvider
from nexus.sdk import Nexus
from nexus.web.templates import HTML
from nexus.providers.base import ProviderError
from nexus.knowledge.base import KnowledgeError
from nexus.core.exceptions import (
    ValidationError as BioNexusValidationError,
    ConfigurationError,
)
from nexus.core.logging import get_correlation_id, set_correlation_id
from nexus.application.app import BioNexus
from nexus.api.analysis_store import AnalysisStore
from nexus.graph.edge import EdgeType
from nexus.api.bionexus_models import (
    BioNexusAnalyzeRequest,
    BioNexusAnalyzeResponse,
    BioNexusSubject,
    BioNexusRankedEntity,
    BioNexusEvidence,
    BioNexusEvidenceResponse,
    BioNexusLiteratureItem,
    BioNexusLiteratureResponse,
    BioNexusStructureItem,
    BioNexusStructureResponse,
    BioNexusHypothesis,
    BioNexusContradiction,
    BioNexusScientificConfidence,
    BioNexusGraphNode,
    BioNexusGraphEdge,
    BioNexusGraphResponse,
)

_api_logger = logging.getLogger("nexus.api")

app = FastAPI(
    title="Nexus Ω — Biological Interpretation Intelligence",
    description="AI-native scientific operating system for biology.",
    version="0.1.0",
)


import hmac
import threading

from nexus.core.config import config
from nexus.api.rate_limiter import RateLimiter


# ── Rate limiting (in-process) ───────────────────────────────────────
# IN-PROCESS fixed-window limiter: suitable for single-process deployment,
# development, and small deployments.  NOT a distributed/global limiter —
# see nexus.api.rate_limiter for the full limitation notes.
_RATE_LIMITER: RateLimiter | None = None
_RATE_LIMITER_LOCK = threading.Lock()


def _get_rate_limiter() -> RateLimiter:
    """Return the process-global rate limiter.

    The limiter is (re)built from ``config.rate_limit`` whenever the
    configured limits change, so configuration updates take effect without
    a restart.  Rebuilding discards prior window state, which is acceptable
    for an in-process limiter (and convenient in tests).
    """
    global _RATE_LIMITER
    with _RATE_LIMITER_LOCK:
        if (
            _RATE_LIMITER is None
            or _RATE_LIMITER.max_requests != config.rate_limit.max_requests
            or _RATE_LIMITER.window_seconds != config.rate_limit.window_seconds
            or _RATE_LIMITER.max_identities != config.rate_limit.max_identities
        ):
            _RATE_LIMITER = RateLimiter(
                max_requests=config.rate_limit.max_requests,
                window_seconds=config.rate_limit.window_seconds,
                max_identities=config.rate_limit.max_identities,
            )
        return _RATE_LIMITER


def _client_ip(request: Request) -> str:
    """Return the client identity source for rate limiting.

    Default is the direct TCP peer.  ``X-Forwarded-For`` is consulted ONLY
    when the direct peer appears in ``config.api.trusted_proxies`` (default
    empty) — an untrusted client therefore cannot mint fresh buckets by
    sending the header.  For a trusted proxy chain the identity is the
    rightmost untrusted hop; if every advertised hop is trusted, the
    leftmost hop (closest to the true client) is used.
    """
    peer = request.client.host if request.client else "unknown"
    trusted = {p.strip() for p in config.api.trusted_proxies if p.strip()}
    if not trusted or peer not in trusted:
        return peer
    xff = request.headers.get("x-forwarded-for")
    if not xff:
        return peer
    hops = [h.strip() for h in xff.split(",") if h.strip()]
    for hop in reversed(hops):
        if hop not in trusted:
            return hop
    return hops[0] if hops else peer


def set_rate_limiter(limiter: RateLimiter | None) -> None:
    """Replace or clear (``None``) the process-global rate limiter.

    Intended for tests: clearing forces recreation from configuration, and
    injecting a limiter with a custom clock enables deterministic
    window-expiry tests without sleeps.
    """
    global _RATE_LIMITER
    with _RATE_LIMITER_LOCK:
        _RATE_LIMITER = limiter


@app.middleware("http")
async def security_headers_and_auth_middleware(request: Request, call_next):
    # Correlation ID handling
    cid = request.headers.get("x-correlation-id") or uuid.uuid4().hex[:12]
    set_correlation_id(cid)

    # Security Headers to attach to every response
    security_headers = {
        "X-Correlation-ID": cid,
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Cache-Control": "no-store",
    }

    # If CORS preflight / OPTIONS, bypass authentication
    if request.method == "OPTIONS":
        response = await call_next(request)
        for k, v in security_headers.items():
            response.headers[k] = v
        return response

    path = request.url.path

    # Unprotected public endpoints
    # - Root UI: /
    # - Health checks: /api/health, /api/bionexus/health
    # - Docs/OpenAPI (if accessed)
    unprotected_paths = {
        "/",
        "/api/health",
        "/api/bionexus/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    # Check if API key authentication is enabled
    configured_key = getattr(config.api, "api_key", "")

    # Populated only when authentication below succeeds with a key.
    provided_key: str | None = None
    if configured_key and path not in unprotected_paths and path.startswith("/api/"):
        # Check Authorization header: Bearer <key> or X-API-Key: <key>
        auth_header = request.headers.get("Authorization")
        if auth_header:
            parts = auth_header.split(" ", 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                provided_key = parts[1].strip()
            else:
                provided_key = auth_header.strip()

        if not provided_key:
            x_api_key = request.headers.get("x-api-key")
            if x_api_key:
                provided_key = x_api_key.strip()

        if not provided_key:
            _api_logger.warning("Missing API key for protected endpoint [%s]: %s", cid, path)
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Authentication required.",
                    "correlation_id": cid,
                },
                headers=security_headers,
            )

        # Constant-time comparison to prevent timing attacks.
        #
        # Compare as UTF-8 *bytes*: ``hmac.compare_digest`` on ``bytes``
        # accepts any content, so unusual headers (obs-text bytes 0x80-0xFF,
        # e.g. mis-decoded UTF-8 in ``X-API-Key``) produce a clean 403 instead
        # of an unhandled TypeError/500.  The comparison remains
        # length-safe and content-independent in timing.
        if not hmac.compare_digest(
            provided_key.encode("utf-8"), configured_key.encode("utf-8")
        ):
            _api_logger.warning("Invalid API key for protected endpoint [%s]: %s", cid, path)
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Forbidden: invalid credentials.",
                    "correlation_id": cid,
                },
                headers=security_headers,
            )

    # ── Rate limiting (Step 4) ───────────────────────────────────────
    # Authentication above determined WHO the caller is; rate limiting
    # determines WHETHER that identity may continue.  This is an IN-PROCESS
    # limiter: suitable for single-process/dev/small deployments only —
    # NOT a distributed/global limiter for multiple workers or servers.
    if (
        config.rate_limit.enabled
        and path not in unprotected_paths
        and path.startswith("/api/")
    ):
        if configured_key and provided_key:
            # Use a SHA-256 fingerprint — the raw key is never stored or logged.
            identity = "key:" + RateLimiter.fingerprint(provided_key)
        else:
            # Dev mode (no API key configured): fall back to client IP.
            # Behind a trusted reverse proxy this resolves the forwarded
            # client (see _client_ip); otherwise the direct peer is used.
            identity = "ip:" + _client_ip(request)
        allowed, retry_after = _get_rate_limiter().check(identity)
        if not allowed:
            _api_logger.warning(
                "Rate limit exceeded [%s]: identity=%s path=%s", cid, identity, path
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded.",
                    "correlation_id": cid,
                },
                headers={**security_headers, "Retry-After": str(retry_after)},
            )

    response = await call_next(request)
    for k, v in security_headers.items():
        response.headers[k] = v
    return response


# ── Centralized exception handlers ─────────────────────────────────────


@app.exception_handler(ProviderError)
async def _handle_provider_error(request: Request, exc: ProviderError) -> JSONResponse:
    correlation_id = get_correlation_id()
    _api_logger.error("ProviderError [%s]: %s", correlation_id, exc, exc_info=True)
    return JSONResponse(
        status_code=502,
        content={
            "error": "Upstream provider request failed.",
            "correlation_id": correlation_id,
        },
        headers={"X-Correlation-ID": correlation_id},
    )


@app.exception_handler(KnowledgeError)
async def _handle_knowledge_error(request: Request, exc: KnowledgeError) -> JSONResponse:
    correlation_id = get_correlation_id()
    _api_logger.error("KnowledgeError [%s]: %s", correlation_id, exc, exc_info=True)
    return JSONResponse(
        status_code=502,
        content={
            "error": "Knowledge source request failed.",
            "correlation_id": correlation_id,
        },
        headers={"X-Correlation-ID": correlation_id},
    )


@app.exception_handler(BioNexusValidationError)
async def _handle_validation_error(request: Request, exc: BioNexusValidationError) -> JSONResponse:
    correlation_id = get_correlation_id()
    _api_logger.error("ValidationError [%s]: %s", correlation_id, exc, exc_info=True)
    return JSONResponse(
        status_code=400,
        content={
            "error": "Input validation failed.",
            "correlation_id": correlation_id,
        },
        headers={"X-Correlation-ID": correlation_id},
    )


@app.exception_handler(ConfigurationError)
async def _handle_configuration_error(request: Request, exc: ConfigurationError) -> JSONResponse:
    correlation_id = get_correlation_id()
    _api_logger.error("ConfigurationError [%s]: %s", correlation_id, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal configuration error.",
            "correlation_id": correlation_id,
        },
        headers={"X-Correlation-ID": correlation_id},
    )


# Process-global Nexus instance. Set via set_nexus() before serving.
_NEXUS: Nexus | None = None


def set_nexus(nexus: Nexus) -> None:
    """Set the process-global Nexus instance."""
    global _NEXUS
    _NEXUS = nexus


def get_nexus() -> Nexus:
    """Get the process-global Nexus instance, creating a default if missing."""
    global _NEXUS
    if _NEXUS is None:
        _NEXUS = Nexus(provider=DummyProvider())
    return _NEXUS


# Process-global BioNexus instance. Set via set_bionexus() before serving.
_BIONEXUS: BioNexus | None = None
_ANALYSIS_STORE = AnalysisStore()


def set_bionexus(bionexus: BioNexus) -> None:
    """Set the process-global BioNexus instance."""
    global _BIONEXUS
    _BIONEXUS = bionexus


def get_bionexus() -> BioNexus:
    """Get the process-global BioNexus instance, creating a default if missing."""
    global _BIONEXUS
    if _BIONEXUS is None:
        _BIONEXUS = BioNexus()
    return _BIONEXUS


# ── Request / response models ────────────────────────────────────────


class InterpretRequest(BaseModel):
    question: str = Field(..., description="The question to ask.")
    agent: str = Field("literature", description="Agent to use.")
    context: dict[str, Any] = Field(default_factory=dict)


class InterpretResponse(BaseModel):
    answer: str
    confidence: float
    provider_model: str | None
    citations: list[str]
    contradictions: list[str]
    recommended_next_step: str | None
    evidence_ids: list[str]


class ValidateRequest(BaseModel):
    claim: str
    context: dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    source_uri: str
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]


class BiokitRequest(BaseModel):
    program: str
    inputs: dict[str, Any]


class BiokitResponse(BaseModel):
    program: str
    outputs: dict[str, Any]
    fingerprint: str


class ReportRequest(BaseModel):
    title: str = "Nexus report"
    questions: list[str] = Field(default_factory=list)


class ReportResponse(BaseModel):
    title: str
    markdown: str
    n_sections: int


class HealthResponse(BaseModel):
    status: str
    version: str
    agents: list[str]
    provider: str


# ── Routes ────────────────────────────────────────────────────────────


@app.get("/")
def root() -> HTMLResponse:
    """Serve the minimal web UI."""
    return HTMLResponse(HTML)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    nx = get_nexus()
    from nexus import __version__

    return HealthResponse(
        status="ok",
        version=__version__,
        agents=nx.list_agents(),
        provider=nx.provider.name,
    )


@app.get("/api/agents", response_model=list[str])
def agents() -> list[str]:
    return get_nexus().list_agents()


@app.post("/api/interpret", response_model=InterpretResponse)
def interpret(req: InterpretRequest) -> InterpretResponse:
    nx = get_nexus()
    try:
        result = nx.interpret(req.question, agent=req.agent, context=req.context or None)
    except (ProviderError, KnowledgeError, BioNexusValidationError, ConfigurationError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    return InterpretResponse(
        answer=result.answer,
        confidence=result.confidence.value,
        provider_model=result.provider_model,
        citations=result.citations,
        contradictions=result.contradictions,
        recommended_next_step=result.recommended_next_step,
        evidence_ids=result.evidence_ids,
    )


@app.post("/api/validate", response_model=InterpretResponse)
def validate(req: ValidateRequest) -> InterpretResponse:
    nx = get_nexus()
    try:
        result = nx.interpret(req.claim, agent="validation", context=req.context or None)
    except (ProviderError, KnowledgeError, BioNexusValidationError, ConfigurationError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    return InterpretResponse(
        answer=result.answer,
        confidence=result.confidence.value,
        provider_model=result.provider_model,
        citations=result.citations,
        contradictions=result.contradictions,
        recommended_next_step=result.recommended_next_step,
        evidence_ids=result.evidence_ids,
    )


@app.post("/api/index")
def index(req: IndexRequest) -> dict[str, Any]:
    nx = get_nexus()
    nx.index_text(req.source_uri, req.title, req.content, req.metadata)
    return {"status": "indexed", "title": req.title, "uri": req.source_uri}


@app.get("/api/search", response_model=SearchResponse)
def search(q: str, limit: int = 5) -> SearchResponse:
    nx = get_nexus()
    results = nx.retriever.retrieve(q, limit=limit)
    return SearchResponse(
        query=q,
        results=[r.to_dict() for r in results],
    )


@app.post("/api/biokit", response_model=BiokitResponse)
def run_biokit(req: BiokitRequest) -> BiokitResponse:
    nx = get_nexus()
    try:
        output = nx.run_biokit(req.program, req.inputs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"BioKit program not found: {req.program}") from exc
    except (ProviderError, KnowledgeError, BioNexusValidationError, ConfigurationError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    return BiokitResponse(
        program=output.program,
        outputs=output.outputs,
        fingerprint=output.fingerprint,
    )


@app.post("/api/report", response_model=ReportResponse)
def generate_report(req: ReportRequest) -> ReportResponse:
    nx = get_nexus()
    interpretations = []
    for q in req.questions:
        try:
            interpretations.append(nx.interpret(q, agent="literature"))
        except Exception:
            continue
    report = nx.generate_report(req.title, interpretations)
    return ReportResponse(
        title=report.title,
        markdown=report.to_markdown(),
        n_sections=len(report.sections),
    )


@app.get("/api/bionexus/health")
def bionexus_health() -> dict[str, Any]:
    """Health check for BioNexus scientific pipeline."""
    from nexus import __version__

    return {
        "status": "ok",
        "version": __version__,
        "pipeline": "BioNexus Scientific Pipeline",
    }


@app.post("/api/bionexus/analyze", response_model=BioNexusAnalyzeResponse)
async def bionexus_analyze(
    req: BioNexusAnalyzeRequest, request: Request
) -> BioNexusAnalyzeResponse:
    """Run an end-to-end scientific analysis via the BioNexus pipeline."""
    cid = request.headers.get("x-correlation-id") or get_correlation_id()
    set_correlation_id(cid)

    bn = get_bionexus()
    try:
        ctx = await bn.analyze_context(req.query, correlation_id=cid)
    except (ProviderError, KnowledgeError, BioNexusValidationError, ConfigurationError):
        raise
    except Exception as exc:
        _api_logger.error("BioNexus analysis failed [%s]: %s", cid, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    analysis_id = _ANALYSIS_STORE.create(ctx)

    subject_model = None
    if ctx.subject:
        subject_model = BioNexusSubject(
            name=ctx.subject.name or ctx.subject.id,
            id=ctx.subject.id,
            type=str(ctx.subject.entity_type),
        )

    summary = None
    reasoning = []
    evidence_strings = []
    rep_conf = None
    if ctx.report and ctx.report.explanation:
        summary = ctx.report.explanation.summary
        reasoning = list(ctx.report.explanation.reasoning)
        evidence_strings = list(ctx.report.explanation.evidence)
        rep_conf = ctx.report.explanation.confidence

    ranked_entities = []
    if ctx.report and ctx.report.ranked_entities:
        ranked_entities = [
            BioNexusRankedEntity(entity_id=r.entity_id, score=r.score)
            for r in ctx.report.ranked_entities
        ]

    hypotheses = []
    if ctx.report and ctx.report.hypotheses:
        hypotheses = [
            BioNexusHypothesis(
                title=h.title,
                description=h.description,
                supporting_entities=list(h.supporting_entities),
                confidence=h.confidence,
                rationale=h.rationale,
            )
            for h in ctx.report.hypotheses
        ]

    contradictions = []
    if ctx.report and ctx.report.contradictions:
        contradictions = [
            BioNexusContradiction(
                topic=c.topic,
                claim_a=c.claim_a,
                claim_b=c.claim_b,
                source_a=c.source_a,
                source_b=c.source_b,
                confidence=c.confidence,
            )
            for c in ctx.report.contradictions
        ]

    sci_conf = None
    if ctx.report and ctx.report.scientific_confidence:
        sc = ctx.report.scientific_confidence
        sci_conf = BioNexusScientificConfidence(
            evidence_quality=sc.evidence_quality,
            source_diversity=sc.source_diversity,
            relationship_support=sc.relationship_support,
            literature_support=sc.literature_support,
            consistency=sc.consistency,
            overall=sc.overall,
        )

    metadata = dict(ctx.report.metadata) if ctx.report else {}
    metadata["correlation_id"] = cid

    return BioNexusAnalyzeResponse(
        analysis_id=analysis_id,
        query=ctx.query,
        subject=subject_model,
        summary=summary,
        reasoning=reasoning,
        evidence=evidence_strings,
        ranked_entities=ranked_entities,
        hypotheses=hypotheses,
        contradictions=contradictions,
        representation_confidence=rep_conf,
        scientific_confidence=sci_conf,
        metadata=metadata,
    )


@app.get("/api/bionexus/graph/{analysis_id}", response_model=BioNexusGraphResponse)
def bionexus_graph(analysis_id: str) -> BioNexusGraphResponse:
    """Retrieve the generated knowledge graph for an analysis."""
    ctx = _ANALYSIS_STORE.get(analysis_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    nodes = []
    edges = []
    subject_id = ctx.subject.id if ctx.subject else None

    if ctx.graph:
        for node in ctx.graph.nodes.values():
            nodes.append(
                BioNexusGraphNode(
                    id=node.id,
                    label=node.label,
                    type=node.type,
                    aliases=list(node.aliases),
                    properties=dict(node.properties),
                    confidence=float(node.confidence),
                )
            )
        for edge in ctx.graph.edges:
            rel_str = edge.relation.value if hasattr(edge.relation, "value") else str(edge.relation)
            edges.append(
                BioNexusGraphEdge(
                    source=edge.source,
                    target=edge.target,
                    relation=rel_str,
                    weight=float(edge.weight),
                    confidence=float(edge.confidence),
                    evidence=list(edge.evidence),
                    properties=dict(edge.properties),
                    directed=bool(edge.directed),
                )
            )

    return BioNexusGraphResponse(
        analysis_id=analysis_id,
        subject_id=subject_id,
        node_count=len(nodes),
        edge_count=len(edges),
        nodes=nodes,
        edges=edges,
    )


@app.get("/api/bionexus/evidence/{analysis_id}", response_model=BioNexusEvidenceResponse)
def bionexus_evidence(analysis_id: str) -> BioNexusEvidenceResponse:
    """Retrieve aggregated evidence for an analysis."""
    ctx = _ANALYSIS_STORE.get(analysis_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    evidence_items = [
        BioNexusEvidence(
            source=e.source,
            identifier=e.identifier,
            title=e.title,
            confidence=float(e.confidence),
            metadata=dict(e.metadata),
        )
        for e in ctx.evidence
    ]

    return BioNexusEvidenceResponse(
        analysis_id=analysis_id,
        count=len(evidence_items),
        evidence=evidence_items,
    )


@app.get("/api/bionexus/literature/{analysis_id}", response_model=BioNexusLiteratureResponse)
def bionexus_literature(analysis_id: str) -> BioNexusLiteratureResponse:
    """Retrieve literature references linked in the analysis."""
    ctx = _ANALYSIS_STORE.get(analysis_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    lit_items: list[BioNexusLiteratureItem] = []
    seen: set[str] = set()

    if ctx.graph:
        for edge in ctx.graph.edges:
            target_node = ctx.graph.get_node(edge.target)
            if target_node and target_node.type.lower() in ("publication", "literature"):
                if target_node.id not in seen:
                    seen.add(target_node.id)
                    lit_items.append(
                        BioNexusLiteratureItem(
                            publication_id=target_node.id,
                            title=target_node.label,
                            relevance=float(edge.weight),
                            confidence=float(edge.confidence),
                            tier=str(edge.properties.get("tier")) if edge.properties.get("tier") else None,
                            evidence=list(edge.evidence),
                        )
                    )
        for node in ctx.graph.nodes.values():
            if node.type.lower() in ("publication", "literature") and node.id not in seen:
                seen.add(node.id)
                lit_items.append(
                    BioNexusLiteratureItem(
                        publication_id=node.id,
                        title=node.label,
                        relevance=float(node.properties.get("relevance", node.confidence)),
                        confidence=float(node.confidence),
                        tier=str(node.properties.get("tier")) if node.properties.get("tier") else None,
                        evidence=list(node.evidence),
                    )
                )

    for e in ctx.evidence:
        if e.source.lower() in ("pubmed", "literature") and e.identifier not in seen:
            seen.add(e.identifier)
            lit_items.append(
                BioNexusLiteratureItem(
                    publication_id=e.identifier,
                    title=e.title,
                    relevance=float(e.confidence),
                    confidence=float(e.confidence),
                    tier=e.metadata.get("tier"),
                    evidence=[f"{e.source}:{e.identifier}"],
                )
            )

    return BioNexusLiteratureResponse(
        analysis_id=analysis_id,
        count=len(lit_items),
        literature=lit_items,
    )


@app.get("/api/bionexus/structures/{analysis_id}", response_model=BioNexusStructureResponse)
def bionexus_structures(analysis_id: str) -> BioNexusStructureResponse:
    """Retrieve 3D macromolecular structures linked in the analysis."""
    ctx = _ANALYSIS_STORE.get(analysis_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    structures: list[BioNexusStructureItem] = []
    seen: set[str] = set()

    if ctx.graph:
        for edge in ctx.graph.edges:
            rel_name = edge.relation.value if hasattr(edge.relation, "value") else str(edge.relation)
            target_node = ctx.graph.get_node(edge.target)
            if rel_name == "HAS_STRUCTURE" or (target_node and target_node.type.lower() in ("structure", "pdb")):
                struct_id = target_node.id if target_node else edge.target
                if struct_id not in seen:
                    seen.add(struct_id)
                    structures.append(
                        BioNexusStructureItem(
                            structure_id=struct_id,
                            title=target_node.label if target_node else struct_id,
                            method=target_node.properties.get("method") if target_node else None,
                            resolution=str(target_node.properties.get("resolution")) if target_node and "resolution" in target_node.properties else None,
                            confidence=float(target_node.confidence if target_node else edge.confidence),
                            evidence=list(edge.evidence) if edge.evidence else (list(target_node.evidence) if target_node else []),
                            properties=dict(target_node.properties) if target_node else {},
                        )
                    )
        for node in ctx.graph.nodes.values():
            if node.type.lower() in ("structure", "pdb") and node.id not in seen:
                seen.add(node.id)
                structures.append(
                    BioNexusStructureItem(
                        structure_id=node.id,
                        title=node.label,
                        method=node.properties.get("method"),
                        resolution=str(node.properties.get("resolution")) if "resolution" in node.properties else None,
                        confidence=float(node.confidence),
                        evidence=list(node.evidence),
                        properties=dict(node.properties),
                    )
                )

    for e in ctx.evidence:
        if e.source.lower() in ("pdb", "structure") and e.identifier not in seen:
            seen.add(e.identifier)
            structures.append(
                BioNexusStructureItem(
                    structure_id=e.identifier,
                    title=e.title,
                    method=e.metadata.get("method"),
                    resolution=e.metadata.get("resolution"),
                    confidence=float(e.confidence),
                    evidence=[f"{e.source}:{e.identifier}"],
                    properties=dict(e.metadata),
                )
            )

    return BioNexusStructureResponse(
        analysis_id=analysis_id,
        count=len(structures),
        structures=structures,
    )


__all__ = [
    "app",
    "get_nexus",
    "set_nexus",
    "get_bionexus",
    "set_bionexus",
]

