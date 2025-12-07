"""
Default constants for COP pipeline.

These can be overridden via:
1. Environment variables (COP_LLM_ENDPOINT, COP_EMBEDDING_MODEL, etc.)
2. cop.yaml build config
3. CLI arguments

Priority: CLI > cop.yaml > environment > defaults
"""

import os
import warnings

# =============================================================================
# ENDPOINTS
# =============================================================================
DEFAULT_LLM_ENDPOINT = os.environ.get("COP_LLM_ENDPOINT", "http://localhost:1234/v1")
DEFAULT_EMBEDDING_ENDPOINT = os.environ.get("COP_EMBEDDING_ENDPOINT", DEFAULT_LLM_ENDPOINT)

# =============================================================================
# MODELS
# =============================================================================
DEFAULT_LLM_MODEL = os.environ.get("COP_LLM_MODEL", "local-model")
DEFAULT_EMBEDDING_MODEL = os.environ.get("COP_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")

# Default API key (most local LLMs don't need one)
DEFAULT_API_KEY = os.environ.get("COP_API_KEY", "not-needed")

# =============================================================================
# CHUNKING DEFAULTS
# =============================================================================
DEFAULT_CHUNK_SIZE = int(os.environ.get("COP_CHUNK_SIZE", "512"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("COP_CHUNK_OVERLAP", "50"))
DEFAULT_MIN_CHUNK_SIZE = int(os.environ.get("COP_MIN_CHUNK_SIZE", "50"))
DEFAULT_CHUNKING_STRATEGY = os.environ.get("COP_CHUNKING_STRATEGY", "semantic")

# =============================================================================
# SYNTHETIC DATA DEFAULTS
# =============================================================================
DEFAULT_SYNTHETIC_SAMPLES = int(os.environ.get("COP_SYNTHETIC_SAMPLES", "10"))
DEFAULT_SYNTHETIC_TEMPERATURE = float(os.environ.get("COP_SYNTHETIC_TEMPERATURE", "0.8"))
DEFAULT_RAG_TOP_K = int(os.environ.get("COP_RAG_TOP_K", "5"))

# =============================================================================
# FAISS AVAILABILITY (auto-detected)
# =============================================================================
try:
    import faiss  # noqa: F401
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# =============================================================================
# DEAD CONFIG SECTIONS (for deprecation warnings)
# =============================================================================
DEAD_CONFIG_SECTIONS = frozenset({
    "compatibility",
    "runtime", 
    "observability",
    "dependencies",
    "dev_dependencies",
})

UNIMPLEMENTED_BUILD_OPTIONS = frozenset({
    "preprocessing",
    "postprocessing",
    "targets",  # The per-platform target configs
})


def warn_dead_config(manifest: dict, verbose: bool = False) -> None:
    """Emit warnings for config sections that aren't implemented."""
    for section in DEAD_CONFIG_SECTIONS:
        if section in manifest:
            if verbose:
                warnings.warn(
                    f"Config section '{section}' is not implemented and will be ignored",
                    UserWarning,
                    stacklevel=2
                )
    
    build = manifest.get("build", {})
    for option in UNIMPLEMENTED_BUILD_OPTIONS:
        if option in build:
            if verbose:
                warnings.warn(
                    f"Build option '{option}' is not implemented and will be ignored",
                    UserWarning,
                    stacklevel=2
                )

