"""
COP CLI - Context-Oriented Programming Command Line Interface

Build, validate, and deploy LLM agents from context modules.
"""

__version__ = "0.1.0"
__author__ = "COP Research Team"

from cop.core.validator import COPValidator, ValidationResult
from cop.core.package import COPPackage

__all__ = [
    "__version__",
    "COPValidator",
    "ValidationResult",
    "COPPackage",
]

