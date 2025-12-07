"""
COP Package - Package loading and manipulation

This module provides functionality to load, parse, and manipulate COP packages.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SystemPrompt:
    """System prompt configuration."""
    source: str
    content: str = ""
    compiled: str = ""
    variables: List[str] = field(default_factory=list)


@dataclass
class Persona:
    """Persona configuration."""
    name: str
    source: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Guardrail:
    """Guardrail configuration."""
    name: str
    source: str
    priority: int = 50
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """Tool definition."""
    name: str
    source: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Knowledge:
    """Knowledge source."""
    name: str
    source: str
    type: str = "static"  # static, structured, dynamic
    content: str = ""


@dataclass
class COPPackage:
    """Represents a loaded COP package."""
    path: Path
    manifest: Dict[str, Any]
    system_prompt: Optional[SystemPrompt] = None
    personas: Dict[str, Persona] = field(default_factory=dict)
    guardrails: List[Guardrail] = field(default_factory=list)
    tools: List[Tool] = field(default_factory=list)
    knowledge: List[Knowledge] = field(default_factory=list)
    
    @property
    def name(self) -> str:
        """Get package name."""
        return self.manifest.get("meta", {}).get("name", self.path.name)
    
    @property
    def version(self) -> str:
        """Get package version."""
        return self.manifest.get("meta", {}).get("version", "0.0.0")
    
    @property
    def description(self) -> str:
        """Get package description."""
        return self.manifest.get("meta", {}).get("description", "")
    
    @property
    def default_persona(self) -> Optional[str]:
        """Get default persona name."""
        return self.manifest.get("context", {}).get("personas", {}).get("default")
    
    @classmethod
    def load(cls, package_path: Path) -> "COPPackage":
        """Load a COP package from disk."""
        package_path = Path(package_path)
        
        if not package_path.exists():
            raise FileNotFoundError(f"Package path does not exist: {package_path}")
        
        # Load manifest
        cop_yaml_path = package_path / "cop.yaml"
        if not cop_yaml_path.exists():
            raise FileNotFoundError(f"Missing cop.yaml: {cop_yaml_path}")
        
        with open(cop_yaml_path, encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        
        if manifest is None:
            raise ValueError("Empty cop.yaml manifest")
        
        package = cls(path=package_path, manifest=manifest)
        
        # Load components
        package._load_system_prompt()
        package._load_personas()
        package._load_guardrails()
        package._load_tools()
        package._load_knowledge()
        
        return package
    
    def _load_system_prompt(self):
        """Load system prompt from file."""
        context = self.manifest.get("context", {})
        system = context.get("system", {})
        
        if "source" in system:
            source_path = self.path / system["source"]
            if source_path.exists():
                with open(source_path, encoding="utf-8") as f:
                    content = f.read()
                
                # Extract variables from template
                variables = re.findall(r'\{\{(\w+)\}\}', content)
                
                self.system_prompt = SystemPrompt(
                    source=system["source"],
                    content=content,
                    variables=list(set(variables))
                )
    
    def _load_personas(self):
        """Load persona files."""
        context = self.manifest.get("context", {})
        personas = context.get("personas", {})
        available = personas.get("available", {})
        
        for name, config in available.items():
            if "source" in config:
                source_path = self.path / config["source"]
                if source_path.exists():
                    with open(source_path, encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    
                    self.personas[name] = Persona(
                        name=name,
                        source=config["source"],
                        data=data
                    )
    
    def _load_guardrails(self):
        """Load guardrail files."""
        context = self.manifest.get("context", {})
        guardrails = context.get("guardrails", [])
        
        for gr_config in guardrails:
            if "source" in gr_config:
                source_path = self.path / gr_config["source"]
                if source_path.exists():
                    with open(source_path, encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    
                    self.guardrails.append(Guardrail(
                        name=gr_config.get("name", source_path.stem),
                        source=gr_config["source"],
                        priority=gr_config.get("priority", 50),
                        data=data
                    ))
        
        # Sort by priority (highest first)
        self.guardrails.sort(key=lambda g: g.priority, reverse=True)
    
    def _load_tools(self):
        """Load tool definitions."""
        context = self.manifest.get("context", {})
        tools = context.get("tools", [])
        
        for tool_config in tools:
            if "source" in tool_config:
                source_path = self.path / tool_config["source"]
                if source_path.exists():
                    with open(source_path, encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    
                    self.tools.append(Tool(
                        name=tool_config.get("name", source_path.stem),
                        source=tool_config["source"],
                        data=data
                    ))
    
    def _load_knowledge(self):
        """Load knowledge files."""
        context = self.manifest.get("context", {})
        knowledge_items = context.get("knowledge", [])
        
        for k_config in knowledge_items:
            if "source" in k_config:
                source_path = self.path / k_config["source"]
                content = ""
                if source_path.exists():
                    with open(source_path, encoding="utf-8") as f:
                        content = f.read()
                
                self.knowledge.append(Knowledge(
                    name=k_config.get("name", source_path.stem),
                    source=k_config["source"],
                    type=k_config.get("type", "static"),
                    content=content
                ))
    
    def compile_prompt(self, variables: Dict[str, Any] = None) -> str:
        """Compile system prompt with variable substitution."""
        if self.system_prompt is None:
            return ""
        
        variables = variables or {}
        content = self.system_prompt.content
        
        # Replace variables
        def replace_var(match):
            var_name = match.group(1)
            default = match.group(2)
            return str(variables.get(var_name, default or f"{{{{UNDEFINED:{var_name}}}}}"))
        
        pattern = r'\{\{(\w+)(?:\|([^}]+))?\}\}'
        compiled = re.sub(pattern, replace_var, content)
        
        self.system_prompt.compiled = compiled
        return compiled
    
    def get_active_persona(self, name: str = None) -> Optional[Persona]:
        """Get persona by name or default."""
        name = name or self.default_persona
        if name and name in self.personas:
            return self.personas[name]
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert package to dictionary representation."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "path": str(self.path),
            "system_prompt": {
                "source": self.system_prompt.source if self.system_prompt else None,
                "variables": self.system_prompt.variables if self.system_prompt else [],
            },
            "personas": {
                name: {"source": p.source, "name": p.name}
                for name, p in self.personas.items()
            },
            "guardrails": [
                {"name": g.name, "source": g.source, "priority": g.priority}
                for g in self.guardrails
            ],
            "tools": [
                {"name": t.name, "source": t.source}
                for t in self.tools
            ],
            "knowledge": [
                {"name": k.name, "source": k.source, "type": k.type}
                for k in self.knowledge
            ],
        }

