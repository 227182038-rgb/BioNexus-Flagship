"""Central configuration for BioNexus."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(slots=True)
class APISettings:
    """External service configuration."""

    pubmed_email: str = ""
    ncbi_api_key: str = ""
    uniprot_api_url: str = "https://rest.uniprot.org"
    pdb_api_url: str = "https://data.rcsb.org"
    api_key: str = ""
    # Direct peers whose X-Forwarded-For may be trusted for rate-limit
    # identity.  EMPTY BY DEFAULT: an untrusted client must not be able to
    # mint fresh rate-limit buckets by sending the header (see
    # nexus.api.app._client_ip).  Behind a reverse proxy, set this to the
    # proxy's address(es), e.g. ["127.0.0.1"].
    trusted_proxies: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CacheSettings:
    """Cache configuration."""

    enabled: bool = True
    directory: Path = Path(".cache")
    ttl_seconds: int = 86400


@dataclass(slots=True)
class GraphSettings:
    """Knowledge graph configuration."""

    directed: bool = True
    default_edge_weight: float = 1.0


@dataclass(slots=True)
class LoggingSettings:
    """Logging configuration."""

    level: str = "INFO"
    file: Path = Path("logs/bionexus.log")


@dataclass(slots=True)
class AISettings:
    """AI-related configuration."""

    model_name: str = "local"
    temperature: float = 0.2
    max_tokens: int = 4096


@dataclass(slots=True)
class RateLimitSettings:
    """In-process API rate-limit configuration.

    This controls a lightweight fixed-window rate limiter that runs
    inside the application process.  It is suitable for single-process
    deployments and development; it is NOT a distributed / global limiter
    for multi-worker or multi-server setups.
    """

    enabled: bool = False
    max_requests: int = 100
    window_seconds: int = 60
    max_identities: int = 10000


@dataclass(slots=True)
class ResourceSettings:
    """Resource caps enforced at the BioKit dispatch boundary.

    In-flight synchronous computations cannot be cancelled mid-run, so these
    caps bound what a NEW invocation may START (they are checked before the
    program executes).  Defaults are generous; raise them if your
    deployment legitimately processes large sequences.
    """

    # Longest allowed single string input (e.g. a DNA sequence).
    max_sequence_length: int = 100_000
    # Cap on O(n·m) dynamic-programming cell counts for pairwise alignment
    # programs (gotoh, needleman_wunsch, smith_waterman): the product of the
    # two longest string inputs must not exceed this.
    max_dp_cells: int = 25_000_000
    # Cap on the canonical-JSON size of the whole inputs payload.
    max_payload_bytes: int = 1_048_576


@dataclass(slots=True)
class BioNexusConfig:
    """Root application configuration."""

    api: APISettings = field(default_factory=APISettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    graph: GraphSettings = field(default_factory=GraphSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    ai: AISettings = field(default_factory=AISettings)
    resources: ResourceSettings = field(default_factory=ResourceSettings)
    rate_limit: RateLimitSettings = field(default_factory=RateLimitSettings)


    @staticmethod
    def _env(key: str, default: str = "") -> str:
        """Read environment variable with optional default.

        This helper isolates env access for easier testing/mocking.
        """
        return os.getenv(key, default)

    def apply_env_overrides(self) -> None:
        """Override configuration values from environment variables.

        Recognized variables:
        - BIO_NEXUS_LOG_LEVEL → logging.level
        - BIO_NEXUS_AI_MODEL_NAME → ai.model_name
        - BIO_NEXUS_AI_TEMPERATURE → ai.temperature (float)
        - BIO_NEXUS_AI_MAX_TOKENS → ai.max_tokens (int)
        - BIO_NEXUS_CACHE_DIR → cache.directory (Path)
        - BIO_NEXUS_OPENAI_BASE_URL → api.base_url (not stored here, handled by ProviderRegistry)
        """
        # Logging
        level = self._env("BIO_NEXUS_LOG_LEVEL")
        if level:
            self.logging.level = level.upper()
        # AI model name
        model = self._env("BIO_NEXUS_AI_MODEL_NAME")
        if model:
            self.ai.model_name = model
        # Temperature
        temp = self._env("BIO_NEXUS_AI_TEMPERATURE")
        if temp:
            try:
                self.ai.temperature = float(temp)
            except ValueError:
                pass
        # Max tokens
        mt = self._env("BIO_NEXUS_AI_MAX_TOKENS")
        if mt:
            try:
                self.ai.max_tokens = int(mt)
            except ValueError:
                pass
        # Cache directory
        cache_dir = self._env("BIO_NEXUS_CACHE_DIR")
        if cache_dir:
            self.cache.directory = Path(cache_dir)
        # BioNexus API security key
        api_key = self._env("BIO_NEXUS_API_KEY")
        if api_key:
            self.api.api_key = api_key
        # Trusted reverse proxies for rate-limit identity (comma-separated)
        trusted = self._env("BIO_NEXUS_API_TRUSTED_PROXIES")
        if trusted:
            self.api.trusted_proxies = [p.strip() for p in trusted.split(",") if p.strip()]
        # Resource caps at the BioKit dispatch boundary
        for env_name, attr in (
            ("BIO_NEXUS_MAX_SEQUENCE_LENGTH", "max_sequence_length"),
            ("BIO_NEXUS_MAX_DP_CELLS", "max_dp_cells"),
            ("BIO_NEXUS_MAX_PAYLOAD_BYTES", "max_payload_bytes"),
        ):
            raw = self._env(env_name)
            if raw:
                try:
                    setattr(self.resources, attr, int(raw))
                except ValueError:
                    pass
        # Rate limiting
        rl_enabled = self._env("BIO_NEXUS_RATE_LIMIT_ENABLED")
        if rl_enabled:
            self.rate_limit.enabled = rl_enabled.lower() in ("1", "true", "yes")
        rl_max = self._env("BIO_NEXUS_RATE_LIMIT_MAX_REQUESTS")
        if rl_max:
            try:
                self.rate_limit.max_requests = int(rl_max)
            except ValueError:
                pass
        rl_window = self._env("BIO_NEXUS_RATE_LIMIT_WINDOW_SECONDS")
        if rl_window:
            try:
                self.rate_limit.window_seconds = int(rl_window)
            except ValueError:
                pass
        rl_max_identities = self._env("BIO_NEXUS_RATE_LIMIT_MAX_IDENTITIES")
        if rl_max_identities:
            try:
                self.rate_limit.max_identities = int(rl_max_identities)
            except ValueError:
                pass

    def validate(self) -> list[str]:
        """Validate configuration values.

        Returns a list of error strings; empty list means the config is valid.
        """
        errors = []
        # Validate log level
        if self.logging.level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
            errors.append(f"Invalid logging level: {self.logging.level}")
        # Validate AI temperature range
        if not (0.0 <= self.ai.temperature <= 2.0):
            errors.append(f"AI temperature out of range [0.0, 2.0]: {self.ai.temperature}")
        # Validate max tokens positive
        if self.ai.max_tokens <= 0:
            errors.append("AI max_tokens must be positive")
        # Validate cache directory is a valid Path
        try:
            _ = self.cache.directory.resolve()
        except Exception:
            errors.append(f"Cache directory is invalid: {self.cache.directory}")
        # Validate rate limit settings
        if self.rate_limit.max_requests <= 0:
            errors.append("rate_limit.max_requests must be positive")
        if self.rate_limit.max_identities <= 0:
            errors.append("rate_limit.max_identities must be positive")
        if self.rate_limit.window_seconds <= 0:
            errors.append("rate_limit.window_seconds must be positive")
        for name in ("max_sequence_length", "max_dp_cells", "max_payload_bytes"):
            if getattr(self.resources, name) <= 0:
                errors.append(f"resources.{name} must be positive")
        return errors


config = BioNexusConfig()


