"""
COP Pipeline - LangGraph-based build pipeline for synthetic data generation
"""

from cop.pipeline.synthetic import SyntheticDataPipeline
from cop.pipeline.constants import (
    DEFAULT_LLM_ENDPOINT,
    DEFAULT_EMBEDDING_ENDPOINT,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_API_KEY,
)

__all__ = [
    "SyntheticDataPipeline",
    "DEFAULT_LLM_ENDPOINT",
    "DEFAULT_EMBEDDING_ENDPOINT",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_API_KEY",
]

