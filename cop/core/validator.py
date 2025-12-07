"""
COP Validator - Schema validation for COP packages

This module provides comprehensive validation for COP packages against
their JSON schemas.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    from jsonschema import Draft7Validator, ValidationError, SchemaError
except ImportError:
    Draft7Validator = None
    ValidationError = Exception
    SchemaError = Exception


class Severity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: Severity
    file_path: str
    message: str
    json_path: str = ""
    schema_path: str = ""
    value: Any = None
    suggestion: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "severity": self.severity.value,
            "file_path": self.file_path,
            "message": self.message,
            "json_path": self.json_path,
            "schema_path": self.schema_path,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of validating a COP package."""
    package_path: str
    issues: list[ValidationIssue] = field(default_factory=list)
    files_validated: int = 0

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    @property
    def is_valid(self) -> bool:
        return not self.has_errors

    def add_error(self, file_path: str, message: str, **kwargs):
        self.issues.append(ValidationIssue(
            severity=Severity.ERROR,
            file_path=file_path,
            message=message,
            **kwargs
        ))

    def add_warning(self, file_path: str, message: str, **kwargs):
        self.issues.append(ValidationIssue(
            severity=Severity.WARNING,
            file_path=file_path,
            message=message,
            **kwargs
        ))

    def add_info(self, file_path: str, message: str, **kwargs):
        self.issues.append(ValidationIssue(
            severity=Severity.INFO,
            file_path=file_path,
            message=message,
            **kwargs
        ))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "package_path": self.package_path,
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "files_validated": self.files_validated,
            "issues": [i.to_dict() for i in self.issues],
        }


class SchemaRegistry:
    """Registry for loading and caching JSON schemas."""

    def __init__(self, schema_dir: Path):
        self.schema_dir = schema_dir
        self._cache: dict[str, dict] = {}
        self._validators: dict[str, Draft7Validator] = {}

    def get_schema(self, schema_name: str) -> Optional[dict]:
        """Load a schema by name (without extension)."""
        if schema_name in self._cache:
            return self._cache[schema_name]

        schema_path = self.schema_dir / f"{schema_name}.schema.json"
        if not schema_path.exists():
            return None

        try:
            with open(schema_path, encoding="utf-8") as f:
                schema = json.load(f)
            self._cache[schema_name] = schema
            return schema
        except (json.JSONDecodeError, IOError):
            return None

    def get_validator(self, schema_name: str) -> Optional[Draft7Validator]:
        """Get a validator for the specified schema."""
        if Draft7Validator is None:
            return None
            
        if schema_name in self._validators:
            return self._validators[schema_name]

        schema = self.get_schema(schema_name)
        if schema is None:
            return None

        try:
            validator = Draft7Validator(schema)
            self._validators[schema_name] = validator
            return validator
        except SchemaError:
            # Invalid schema definition
            return None
        except (TypeError, ValueError):
            # Schema data type issues
            return None


def format_json_path(path: list) -> str:
    """Format a JSON path from a list of path elements."""
    if not path:
        return "$"

    parts = ["$"]
    for element in path:
        if isinstance(element, int):
            parts.append(f"[{element}]")
        else:
            parts.append(f".{element}")

    return "".join(parts)


def get_suggestion_for_error(error: ValidationError) -> str:
    """Generate a helpful suggestion based on the error type."""
    error_type = error.validator

    suggestions = {
        "required": f"Add the missing required field: {error.message}",
        "type": "Change the value to the correct type",
        "enum": f"Use one of the allowed values: {error.schema.get('enum', [])}",
        "pattern": f"Value must match the pattern: {error.schema.get('pattern', '')}",
        "minimum": f"Value must be at least {error.schema.get('minimum', 'N/A')}",
        "maximum": f"Value must be at most {error.schema.get('maximum', 'N/A')}",
        "minLength": f"Value must have at least {error.schema.get('minLength', 'N/A')} characters",
        "maxLength": f"Value must have at most {error.schema.get('maxLength', 'N/A')} characters",
        "additionalProperties": "Remove the unexpected property or update the schema",
        "format": f"Value must be a valid {error.schema.get('format', 'format')}",
    }

    return suggestions.get(error_type, "")


class COPValidator:
    """Main validator for COP packages."""

    COMPONENT_SCHEMAS = {
        "personas": "persona",
        "guardrails": "guardrail",
        "tools": "tool",
        "tests/behavioral": "test",
        "tests/safety": "test",
        "tests/unit": "test",
        "tests/regression": "test",
        "tests/code-quality": "test",
    }

    def __init__(self, schema_dir: Optional[Path] = None):
        if schema_dir is None:
            # Default to the schemas directory relative to package root
            schema_dir = Path(__file__).parent.parent.parent / "schemas"

        self.schema_registry = SchemaRegistry(schema_dir)

    def validate_package(self, package_path: Path) -> ValidationResult:
        """Validate an entire COP package."""
        result = ValidationResult(package_path=str(package_path))

        # Check if package directory exists
        if not package_path.exists():
            result.add_error(
                str(package_path),
                f"Package directory does not exist: {package_path}"
            )
            return result

        if not package_path.is_dir():
            result.add_error(
                str(package_path),
                f"Path is not a directory: {package_path}"
            )
            return result

        # Find and validate cop.yaml
        cop_yaml_path = package_path / "cop.yaml"
        if not cop_yaml_path.exists():
            result.add_error(
                str(cop_yaml_path),
                "Missing required cop.yaml manifest file",
                suggestion="Create a cop.yaml file with at least 'meta' and 'context' sections"
            )
            return result

        # Load and validate cop.yaml
        manifest = self._validate_manifest(cop_yaml_path, result)
        if manifest is None:
            return result

        result.files_validated += 1

        # Validate referenced component files
        self._validate_referenced_files(package_path, manifest, result)

        return result

    def _validate_manifest(self, manifest_path: Path, result: ValidationResult) -> Optional[dict]:
        """Validate the cop.yaml manifest file."""
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = yaml.safe_load(f)
        except yaml.YAMLError as e:
            result.add_error(
                str(manifest_path),
                f"Invalid YAML syntax: {e}",
                suggestion="Check YAML syntax - ensure proper indentation"
            )
            return None
        except IOError as e:
            result.add_error(
                str(manifest_path),
                f"Could not read file: {e}"
            )
            return None

        if manifest is None:
            result.add_error(
                str(manifest_path),
                "Manifest file is empty",
                suggestion="Add required 'meta' and 'context' sections"
            )
            return None

        # Validate against schema
        validator = self.schema_registry.get_validator("cop-manifest")
        if validator is None:
            result.add_warning(
                str(manifest_path),
                "Could not load cop-manifest schema for validation"
            )
            return manifest

        errors = list(validator.iter_errors(manifest))
        for error in errors:
            json_path = format_json_path(list(error.absolute_path))
            result.add_error(
                str(manifest_path),
                error.message,
                json_path=json_path,
                schema_path=format_json_path(list(error.absolute_schema_path)),
                value=error.instance if not isinstance(error.instance, dict) else None,
                suggestion=get_suggestion_for_error(error)
            )

        return manifest

    def _validate_referenced_files(
        self, package_path: Path, manifest: dict, result: ValidationResult
    ):
        """Validate all files referenced in the manifest."""
        context = manifest.get("context", {})

        # Validate system prompt file exists
        system = context.get("system", {})
        if "source" in system:
            source_path = package_path / system["source"]
            if not source_path.exists():
                result.add_error(
                    str(source_path),
                    f"System prompt file not found: {system['source']}",
                    json_path="$.context.system.source"
                )
            else:
                result.files_validated += 1

        # Validate personas
        personas = context.get("personas", {})
        available = personas.get("available", {})
        for persona_name, persona_config in available.items():
            if "source" in persona_config:
                self._validate_component_file(
                    package_path,
                    persona_config["source"],
                    "persona",
                    f"$.context.personas.available.{persona_name}.source",
                    result
                )

        # Validate knowledge files
        knowledge_items = context.get("knowledge", [])
        for i, item in enumerate(knowledge_items):
            if "source" in item:
                source_path = package_path / item["source"]
                if not source_path.exists():
                    result.add_error(
                        str(source_path),
                        f"Knowledge file not found: {item['source']}",
                        json_path=f"$.context.knowledge[{i}].source"
                    )
                else:
                    result.files_validated += 1

        # Validate guardrails
        guardrails = context.get("guardrails", [])
        for i, guardrail in enumerate(guardrails):
            if "source" in guardrail:
                self._validate_component_file(
                    package_path,
                    guardrail["source"],
                    "guardrail",
                    f"$.context.guardrails[{i}].source",
                    result
                )

        # Validate tools
        tools = context.get("tools", [])
        for i, tool in enumerate(tools):
            if "source" in tool:
                self._validate_component_file(
                    package_path,
                    tool["source"],
                    "tool",
                    f"$.context.tools[{i}].source",
                    result
                )

    def _validate_component_file(
        self,
        package_path: Path,
        source: str,
        schema_name: str,
        json_path: str,
        result: ValidationResult
    ):
        """Validate a component file (persona, guardrail, tool, etc.)."""
        file_path = package_path / source

        if not file_path.exists():
            result.add_error(
                str(file_path),
                f"Referenced file not found: {source}",
                json_path=json_path
            )
            return

        result.files_validated += 1

        # Load and validate the file
        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            result.add_error(
                str(file_path),
                f"Invalid YAML syntax: {e}"
            )
            return
        except IOError as e:
            result.add_error(
                str(file_path),
                f"Could not read file: {e}"
            )
            return

        if data is None:
            result.add_error(
                str(file_path),
                "File is empty or contains only null",
                suggestion="Add the required fields for this component type"
            )
            return

        # Validate against schema
        validator = self.schema_registry.get_validator(schema_name)
        if validator is None:
            result.add_warning(
                str(file_path),
                f"Could not load {schema_name} schema for validation"
            )
            return

        errors = list(validator.iter_errors(data))
        for error in errors:
            error_json_path = format_json_path(list(error.absolute_path))
            result.add_error(
                str(file_path),
                error.message,
                json_path=error_json_path,
                schema_path=format_json_path(list(error.absolute_schema_path)),
                value=error.instance if not isinstance(error.instance, dict) else None,
                suggestion=get_suggestion_for_error(error)
            )

