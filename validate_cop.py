#!/usr/bin/env python3
"""
COP Schema Validator

A comprehensive validation tool for Context-Oriented Programming (COP) configurations.
Validates cop.yaml manifests and all referenced component files (personas, guardrails,
tools, and tests) against their respective JSON schemas.

Usage:
    python validate_cop.py <path_to_cop_package>
    python validate_cop.py examples/customer-support-agent
    python validate_cop.py --all  # Validate all example packages
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

try:
    import jsonschema
    from jsonschema import Draft7Validator, ValidationError
except ImportError:
    print("Error: jsonschema is required. Install with: pip install jsonschema")
    sys.exit(1)


class Severity(Enum):
    """Severity levels for validation errors."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Colors:
    """ANSI color codes for terminal output."""
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        """Disable colors (for non-terminal output)."""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.MAGENTA = ""
        cls.CYAN = ""
        cls.WHITE = ""
        cls.BOLD = ""
        cls.RESET = ""


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

    def format(self, show_details: bool = True) -> str:
        """Format the issue for display."""
        severity_colors = {
            Severity.ERROR: Colors.RED,
            Severity.WARNING: Colors.YELLOW,
            Severity.INFO: Colors.BLUE,
        }
        color = severity_colors.get(self.severity, Colors.WHITE)
        
        lines = []
        
        # Main message
        lines.append(
            f"{color}{Colors.BOLD}[{self.severity.value.upper()}]{Colors.RESET} "
            f"{Colors.CYAN}{self.file_path}{Colors.RESET}"
        )
        
        if self.json_path:
            lines.append(f"  {Colors.MAGENTA}Path:{Colors.RESET} {self.json_path}")
        
        lines.append(f"  {Colors.WHITE}{self.message}{Colors.RESET}")
        
        if show_details:
            if self.value is not None:
                value_str = str(self.value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                lines.append(f"  {Colors.YELLOW}Value:{Colors.RESET} {value_str}")
            
            if self.schema_path:
                lines.append(f"  {Colors.BLUE}Schema:{Colors.RESET} {self.schema_path}")
        
        if self.suggestion:
            lines.append(f"  {Colors.GREEN}Suggestion:{Colors.RESET} {self.suggestion}")
        
        return "\n".join(lines)


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
            with open(schema_path) as f:
                schema = json.load(f)
            self._cache[schema_name] = schema
            return schema
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading schema {schema_path}: {e}")
            return None
    
    def get_validator(self, schema_name: str) -> Optional[Draft7Validator]:
        """Get a validator for the specified schema."""
        if schema_name in self._validators:
            return self._validators[schema_name]
        
        schema = self.get_schema(schema_name)
        if schema is None:
            return None
        
        try:
            validator = Draft7Validator(schema)
            self._validators[schema_name] = validator
            return validator
        except Exception as e:
            print(f"Error creating validator for {schema_name}: {e}")
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
        "type": f"Change the value to the correct type",
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
    
    # Mapping of COP component types to their schemas
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
            # Default to the schemas directory relative to this script
            schema_dir = Path(__file__).parent / "schemas"
        
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
            return result  # Stop if manifest is invalid
        
        result.files_validated += 1
        
        # Validate referenced component files
        self._validate_referenced_files(package_path, manifest, result)
        
        return result
    
    def _validate_manifest(self, manifest_path: Path, result: ValidationResult) -> Optional[dict]:
        """Validate the cop.yaml manifest file."""
        try:
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
        except yaml.YAMLError as e:
            result.add_error(
                str(manifest_path),
                f"Invalid YAML syntax: {e}",
                suggestion="Check YAML syntax - ensure proper indentation and formatting"
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
    
    def _validate_referenced_files(self, package_path: Path, manifest: dict, result: ValidationResult):
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
        
        # Validate test suites
        evaluation = manifest.get("evaluation", {})
        test_suites = evaluation.get("test_suites", [])
        for i, suite in enumerate(test_suites):
            if "path" in suite:
                test_path = package_path / suite["path"]
                if test_path.is_dir():
                    self._validate_test_directory(test_path, f"$.evaluation.test_suites[{i}].path", result)
                elif not test_path.exists():
                    result.add_warning(
                        str(test_path),
                        f"Test suite path not found: {suite['path']}",
                        json_path=f"$.evaluation.test_suites[{i}].path"
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
            with open(file_path) as f:
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
    
    def _validate_test_directory(self, test_dir: Path, json_path: str, result: ValidationResult):
        """Validate all test files in a directory."""
        if not test_dir.exists():
            return
        
        for yaml_file in test_dir.glob("*.yaml"):
            self._validate_component_file(
                test_dir.parent.parent,  # Go up to package root
                str(yaml_file.relative_to(test_dir.parent.parent)),
                "test",
                json_path,
                result
            )
        
        for yml_file in test_dir.glob("*.yml"):
            self._validate_component_file(
                test_dir.parent.parent,
                str(yml_file.relative_to(test_dir.parent.parent)),
                "test",
                json_path,
                result
            )
    
    def validate_file(self, file_path: Path, schema_name: str) -> ValidationResult:
        """Validate a single file against a specific schema."""
        result = ValidationResult(package_path=str(file_path))
        
        if not file_path.exists():
            result.add_error(
                str(file_path),
                f"File not found: {file_path}"
            )
            return result
        
        try:
            with open(file_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            result.add_error(
                str(file_path),
                f"Invalid YAML syntax: {e}"
            )
            return result
        
        if data is None:
            result.add_error(
                str(file_path),
                "File is empty"
            )
            return result
        
        result.files_validated = 1
        
        validator = self.schema_registry.get_validator(schema_name)
        if validator is None:
            result.add_error(
                str(file_path),
                f"Schema not found: {schema_name}"
            )
            return result
        
        errors = list(validator.iter_errors(data))
        for error in errors:
            json_path = format_json_path(list(error.absolute_path))
            result.add_error(
                str(file_path),
                error.message,
                json_path=json_path,
                schema_path=format_json_path(list(error.absolute_schema_path)),
                value=error.instance if not isinstance(error.instance, dict) else None,
                suggestion=get_suggestion_for_error(error)
            )
        
        return result


def print_result(result: ValidationResult, verbose: bool = False):
    """Print validation result to console."""
    print()
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}Validation Results: {Colors.CYAN}{result.package_path}{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print()
    
    if not result.issues:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All validations passed!{Colors.RESET}")
        print(f"  Files validated: {result.files_validated}")
    else:
        # Group issues by file
        issues_by_file: dict[str, list[ValidationIssue]] = {}
        for issue in result.issues:
            if issue.file_path not in issues_by_file:
                issues_by_file[issue.file_path] = []
            issues_by_file[issue.file_path].append(issue)
        
        for file_path, issues in issues_by_file.items():
            print(f"{Colors.BOLD}File: {Colors.CYAN}{file_path}{Colors.RESET}")
            print()
            for issue in issues:
                print(issue.format(show_details=verbose))
                print()
        
        print(f"{Colors.BOLD}{'─' * 60}{Colors.RESET}")
        print(f"Summary: {Colors.RED}{result.error_count} error(s){Colors.RESET}, "
              f"{Colors.YELLOW}{result.warning_count} warning(s){Colors.RESET}")
        print(f"Files validated: {result.files_validated}")
    
    print()


def find_cop_packages(root_dir: Path) -> list[Path]:
    """Find all directories containing cop.yaml."""
    packages = []
    for cop_yaml in root_dir.rglob("cop.yaml"):
        packages.append(cop_yaml.parent)
    return packages


def main():
    parser = argparse.ArgumentParser(
        description="Validate Context-Oriented Programming (COP) packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s examples/customer-support-agent
  %(prog)s --all
  %(prog)s --file examples/customer-support-agent/personas/professional.yaml --schema persona
  %(prog)s examples/customer-support-agent --verbose
        """
    )
    
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a COP package directory"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all COP packages in the examples directory"
    )
    parser.add_argument(
        "--file",
        help="Validate a single file"
    )
    parser.add_argument(
        "--schema",
        choices=["cop-manifest", "persona", "guardrail", "tool", "test"],
        help="Schema to validate against (required with --file)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed error information"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    parser.add_argument(
        "--schema-dir",
        help="Custom schema directory path"
    )
    
    args = parser.parse_args()
    
    # Disable colors if requested or not a terminal
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    
    # Determine schema directory
    schema_dir = Path(args.schema_dir) if args.schema_dir else None
    validator = COPValidator(schema_dir)
    
    all_passed = True
    
    if args.file:
        if not args.schema:
            print(f"{Colors.RED}Error: --schema is required when using --file{Colors.RESET}")
            sys.exit(2)
        
        result = validator.validate_file(Path(args.file), args.schema)
        print_result(result, args.verbose)
        all_passed = not result.has_errors
    
    elif args.all:
        # Find all packages in examples directory
        examples_dir = Path(__file__).parent / "examples"
        if not examples_dir.exists():
            print(f"{Colors.RED}Error: examples directory not found{Colors.RESET}")
            sys.exit(1)
        
        packages = find_cop_packages(examples_dir)
        if not packages:
            print(f"{Colors.YELLOW}No COP packages found in {examples_dir}{Colors.RESET}")
            sys.exit(0)
        
        print(f"{Colors.BOLD}Found {len(packages)} COP package(s) to validate{Colors.RESET}")
        
        for package_path in packages:
            result = validator.validate_package(package_path)
            print_result(result, args.verbose)
            if result.has_errors:
                all_passed = False
    
    elif args.path:
        package_path = Path(args.path)
        result = validator.validate_package(package_path)
        print_result(result, args.verbose)
        all_passed = not result.has_errors
    
    else:
        parser.print_help()
        sys.exit(2)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
