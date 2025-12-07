"""
Dependency Resolver for COP Packages

Recursively parses cop.yaml and builds a complete import tree
of all referenced files (prompts, personas, guardrails, knowledge, tools, tests).
"""

import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum
import yaml


class ContentType(Enum):
    """Types of content in a COP package."""
    MANIFEST = "manifest"
    PROMPT = "prompt"
    PERSONA = "persona"
    GUARDRAIL = "guardrail"
    KNOWLEDGE = "knowledge"
    TOOL = "tool"
    TEST = "test"
    SCHEMA = "schema"
    UNKNOWN = "unknown"


@dataclass
class ResolvedFile:
    """A resolved file in the import tree."""
    path: Path
    relative_path: str
    content_type: ContentType
    content: str
    parsed: Optional[Any] = None  # Parsed YAML/JSON if applicable
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    @property
    def extension(self) -> str:
        return self.path.suffix.lower()
    
    @property
    def is_markdown(self) -> bool:
        return self.extension in ['.md', '.markdown']
    
    @property
    def is_yaml(self) -> bool:
        return self.extension in ['.yaml', '.yml']
    
    @property
    def is_json(self) -> bool:
        return self.extension == '.json'


@dataclass
class ImportTree:
    """Complete import tree for a COP package."""
    root: Path
    manifest: ResolvedFile
    files: Dict[str, ResolvedFile] = field(default_factory=dict)
    
    def get_by_type(self, content_type: ContentType) -> List[ResolvedFile]:
        """Get all files of a specific type."""
        return [f for f in self.files.values() if f.content_type == content_type]
    
    def get_all_content(self) -> List[ResolvedFile]:
        """Get all resolved files including manifest."""
        return [self.manifest] + list(self.files.values())
    
    @property
    def total_files(self) -> int:
        return len(self.files) + 1  # +1 for manifest
    
    def summary(self) -> Dict[str, int]:
        """Get count of files by type."""
        counts = {}
        for f in self.files.values():
            key = f.content_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts


class DependencyResolver:
    """
    Resolves all dependencies in a COP package by recursively
    parsing the manifest and loading all referenced files.
    """
    
    def __init__(self, package_path: Path, verbose: bool = False):
        self.package_path = Path(package_path)
        self.manifest_path = self.package_path / "cop.yaml"
        self._resolved: Set[str] = set()
        self.verbose = verbose
        
    def resolve(self) -> ImportTree:
        """
        Resolve the complete import tree starting from cop.yaml.
        
        Returns:
            ImportTree with all resolved files and their content.
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        # Load manifest
        manifest_content = self._read_file(self.manifest_path)
        manifest_parsed = yaml.safe_load(manifest_content)
        
        manifest = ResolvedFile(
            path=self.manifest_path,
            relative_path="cop.yaml",
            content_type=ContentType.MANIFEST,
            content=manifest_content,
            parsed=manifest_parsed,
            metadata={
                "name": manifest_parsed.get("meta", {}).get("name", "unknown"),
                "version": manifest_parsed.get("meta", {}).get("version", "0.0.0"),
            }
        )
        
        tree = ImportTree(
            root=self.package_path,
            manifest=manifest
        )
        
        # Resolve all dependencies
        self._resolve_context(manifest_parsed.get("context", {}), tree)
        self._resolve_tests(manifest_parsed.get("evaluation", {}), tree)
        self._resolve_schemas(manifest_parsed, tree)
        
        return tree
    
    def _resolve_context(self, context: Dict, tree: ImportTree) -> None:
        """Resolve all context-related files."""
        
        # System prompt
        if "system" in context:
            system = context["system"]
            if "source" in system:
                self._add_file(
                    system["source"],
                    ContentType.PROMPT,
                    tree,
                    metadata={"variables": system.get("variables", {})}
                )
        
        # Personas
        if "personas" in context:
            personas = context["personas"]
            available = personas.get("available", {})
            for name, persona_config in available.items():
                if "source" in persona_config:
                    self._add_file(
                        persona_config["source"],
                        ContentType.PERSONA,
                        tree,
                        metadata={"persona_name": name, "is_default": name == personas.get("default")}
                    )
        
        # Knowledge
        if "knowledge" in context:
            for knowledge in context["knowledge"]:
                if "source" in knowledge:
                    self._add_file(
                        knowledge["source"],
                        ContentType.KNOWLEDGE,
                        tree,
                        metadata={
                            "name": knowledge.get("name"),
                            "type": knowledge.get("type", "static"),
                            "description": knowledge.get("description", "")
                        }
                    )
                    # Also resolve schema if present
                    if "schema" in knowledge:
                        self._add_file(
                            knowledge["schema"],
                            ContentType.SCHEMA,
                            tree,
                            metadata={"for": knowledge.get("name")}
                        )
        
        # Guardrails
        if "guardrails" in context:
            for guardrail in context["guardrails"]:
                if "source" in guardrail:
                    self._add_file(
                        guardrail["source"],
                        ContentType.GUARDRAIL,
                        tree,
                        metadata={
                            "name": guardrail.get("name"),
                            "priority": guardrail.get("priority", 50),
                            "description": guardrail.get("description", "")
                        }
                    )
        
        # Tools
        if "tools" in context:
            for tool in context["tools"]:
                if "source" in tool:
                    self._add_file(
                        tool["source"],
                        ContentType.TOOL,
                        tree,
                        metadata={
                            "name": tool.get("name"),
                            "requires_approval": tool.get("requires_approval", False),
                            "description": tool.get("description", "")
                        }
                    )
    
    def _resolve_tests(self, evaluation: Dict, tree: ImportTree) -> None:
        """Resolve all test suite files."""
        test_suites = evaluation.get("test_suites", [])
        
        for suite in test_suites:
            if "path" in suite:
                test_path = self.package_path / suite["path"]
                if test_path.exists():
                    if test_path.is_dir():
                        # Recursively find all test files
                        for test_file in test_path.rglob("*.yaml"):
                            rel_path = test_file.relative_to(self.package_path)
                            self._add_file(
                                str(rel_path),
                                ContentType.TEST,
                                tree,
                                metadata={
                                    "suite_name": suite.get("name"),
                                    "suite_type": suite.get("type"),
                                    "description": suite.get("description", "")
                                }
                            )
                        for test_file in test_path.rglob("*.yml"):
                            rel_path = test_file.relative_to(self.package_path)
                            self._add_file(
                                str(rel_path),
                                ContentType.TEST,
                                tree,
                                metadata={
                                    "suite_name": suite.get("name"),
                                    "suite_type": suite.get("type"),
                                    "description": suite.get("description", "")
                                }
                            )
                    else:
                        self._add_file(
                            suite["path"],
                            ContentType.TEST,
                            tree,
                            metadata={
                                "suite_name": suite.get("name"),
                                "suite_type": suite.get("type"),
                                "description": suite.get("description", "")
                            }
                        )
    
    def _resolve_schemas(self, manifest: Dict, tree: ImportTree) -> None:
        """Resolve any schema files referenced in the manifest."""
        # Check for schema references in various places
        def find_schema_refs(obj: Any, path: str = "") -> List[str]:
            refs = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "schema" and isinstance(value, str) and value.endswith(".json"):
                        refs.append(value)
                    else:
                        refs.extend(find_schema_refs(value, f"{path}.{key}"))
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    refs.extend(find_schema_refs(item, f"{path}[{i}]"))
            return refs
        
        for schema_path in find_schema_refs(manifest):
            if schema_path not in self._resolved:
                self._add_file(schema_path, ContentType.SCHEMA, tree)
    
    def _add_file(
        self,
        relative_path: str,
        content_type: ContentType,
        tree: ImportTree,
        metadata: Optional[Dict] = None
    ) -> Optional[ResolvedFile]:
        """Add a file to the import tree."""
        # Normalize path
        if relative_path.startswith("./"):
            relative_path = relative_path[2:]
        
        # Skip if already resolved
        if relative_path in self._resolved:
            return tree.files.get(relative_path)
        
        full_path = self.package_path / relative_path
        
        if not full_path.exists():
            # File doesn't exist - skip silently or log warning
            return None
        
        try:
            content = self._read_file(full_path)
            parsed = None
            
            # Parse structured files
            if full_path.suffix.lower() in ['.yaml', '.yml']:
                parsed = yaml.safe_load(content)
            elif full_path.suffix.lower() == '.json':
                parsed = json.loads(content)
            
            resolved = ResolvedFile(
                path=full_path,
                relative_path=relative_path,
                content_type=content_type,
                content=content,
                parsed=parsed,
                metadata=metadata or {}
            )
            
            tree.files[relative_path] = resolved
            self._resolved.add(relative_path)
            
            return resolved
            
        except (IOError, OSError, yaml.YAMLError, json.JSONDecodeError) as e:
            # Non-fatal: log warning and continue without this file
            warnings.warn(f"Could not load {relative_path}: {e}", stacklevel=2)
            return None
    
    def _read_file(self, path: Path) -> str:
        """Read file content with proper encoding."""
        return path.read_text(encoding="utf-8")


def resolve_package(package_path: str | Path) -> ImportTree:
    """
    Convenience function to resolve a COP package.
    
    Args:
        package_path: Path to the COP package directory
        
    Returns:
        ImportTree with all resolved dependencies
    """
    resolver = DependencyResolver(Path(package_path))
    return resolver.resolve()

