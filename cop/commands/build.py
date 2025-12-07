"""
COP Build Command - Build COP packages into deployable artifacts

Refactored with proper stage abstraction because copy-pasting the same
pattern 8 times is what amateurs do.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from openai import APIError, APIConnectionError, RateLimitError
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID

from cop.core.package import COPPackage
from cop.core.validator import COPValidator
from cop.pipeline.constants import DEFAULT_LLM_ENDPOINT, DEFAULT_LLM_MODEL, warn_dead_config


# =============================================================================
# Stage Progress Weights (no more magic numbers scattered like confetti)
# =============================================================================

class StageWeight(Enum):
    """Progress weights for each build stage. Total must equal 100."""
    VALIDATION = 15
    LOAD = 15
    COMPILE = 15
    LINKING = 0  # Included in synthetic weight when enabled
    SYNTHETIC = 15
    DATASET = 15
    EMBEDDINGS = 0  # Included in config weight
    CONFIG = 10
    MANIFEST = 10
    PACKAGE = 5
    
    # Special case for dataset_only mode
    DATASET_ONLY_REMAINING = 40


class StageStatus(Enum):
    """Possible statuses for a build stage."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    DISABLED = "disabled"
    PASSED = "passed"  # For validation


# =============================================================================
# Build Context - Shared state across all stages
# =============================================================================

@dataclass
class BuildContext:
    """Shared context passed through all build stages."""
    
    # Input configuration
    package_path: Path
    output_dir: Path
    skip_validation: bool = False
    skip_synthetic: bool = False
    dataset_only: bool = False
    enable_linking: bool = False
    verbose: bool = False
    lm_studio_url: Optional[str] = None
    local_model: Optional[str] = None
    
    # Build configuration (from cop.yaml)
    build_config: dict = field(default_factory=dict)
    
    # Console and progress
    console: Console = field(default_factory=Console)
    progress: Optional[Progress] = None
    progress_task: Optional[TaskID] = None
    
    # Artifacts produced during build
    package: Optional[COPPackage] = None
    compiled_prompt: Optional[str] = None
    variables: dict = field(default_factory=dict)
    linking_result: Any = None
    vector_index: Any = None
    linker: Any = None  # Store linker for reuse
    synthetic_data: list = field(default_factory=list)
    context_bundle: dict = field(default_factory=dict)
    dataset_path: Optional[Path] = None
    artifact_path: Optional[Path] = None
    
    # Build state tracking
    build_state: dict = field(default_factory=lambda: {
        "stages": {},
        "errors": [],
        "warnings": [],
    })
    
    def add_warning(self, message: str) -> None:
        """Add a warning to build state."""
        self.build_state["warnings"].append(message)
    
    def add_error(self, message: str) -> None:
        """Add an error to build state."""
        self.build_state["errors"].append(message)
    
    def set_stage_result(self, name: str, status: StageStatus, **extra: Any) -> None:
        """Record the result of a stage."""
        self.build_state["stages"][name] = {
            "status": status.value,
            **extra
        }
    
    def update_progress(self, description: str = None, advance: int = 0) -> None:
        """Update progress bar."""
        if self.progress and self.progress_task is not None:
            kwargs = {}
            if description:
                kwargs["description"] = description
            if advance:
                kwargs["advance"] = advance
            self.progress.update(self.progress_task, **kwargs)


# =============================================================================
# Stage Result
# =============================================================================

@dataclass
class StageResult:
    """Result of executing a build stage."""
    success: bool
    status: StageStatus = StageStatus.SUCCESS
    message: Optional[str] = None
    data: dict = field(default_factory=dict)
    
    @classmethod
    def ok(cls, **data: Any) -> StageResult:
        """Create a successful result."""
        return cls(success=True, status=StageStatus.SUCCESS, data=data)
    
    @classmethod
    def skipped(cls, reason: str = None) -> StageResult:
        """Create a skipped result."""
        return cls(success=True, status=StageStatus.SKIPPED, message=reason)
    
    @classmethod
    def disabled(cls) -> StageResult:
        """Create a disabled result."""
        return cls(success=True, status=StageStatus.DISABLED)
    
    @classmethod
    def failed(cls, message: str, **data: Any) -> StageResult:
        """Create a failed result."""
        return cls(success=False, status=StageStatus.FAILED, message=message, data=data)


# =============================================================================
# Build Stage Base Class
# =============================================================================

class BuildStage(ABC):
    """Base class for build stages."""
    
    name: str = "unnamed"
    description: str = "Processing..."
    weight: int = 0
    
    @abstractmethod
    def execute(self, ctx: BuildContext) -> StageResult:
        """Execute the stage. Override in subclasses."""
        pass
    
    def should_skip(self, ctx: BuildContext) -> Optional[str]:
        """Return skip reason if stage should be skipped, None otherwise."""
        return None
    
    def run(self, ctx: BuildContext) -> StageResult:
        """Run the stage with progress tracking and error handling."""
        # Check if should skip
        skip_reason = self.should_skip(ctx)
        if skip_reason:
            ctx.update_progress(advance=self.weight)
            ctx.set_stage_result(self.name, StageStatus.SKIPPED, reason=skip_reason)
            return StageResult.skipped(skip_reason)
        
        # Update progress description
        ctx.update_progress(description=f"[cyan]{self.description}")
        
        # Execute the stage
        try:
            result = self.execute(ctx)
        except Exception as e:
            result = StageResult.failed(str(e))
        
        # Record result and advance progress
        ctx.set_stage_result(
            self.name,
            result.status,
            **result.data,
            **({"reason": result.message} if result.message and not result.success else {})
        )
        ctx.update_progress(advance=self.weight)
        
        return result


# =============================================================================
# Concrete Build Stages
# =============================================================================

class ValidationStage(BuildStage):
    name = "validation"
    description = "Validating package..."
    weight = StageWeight.VALIDATION.value
    
    def should_skip(self, ctx: BuildContext) -> Optional[str]:
        if ctx.skip_validation:
            return "Validation skipped by user"
        return None
    
    def execute(self, ctx: BuildContext) -> StageResult:
        validator = COPValidator()
        result = validator.validate_package(ctx.package_path)
        
        if result.has_errors:
            ctx.console.print()
            ctx.console.print("[red]✗ Validation failed[/]")
            for issue in result.issues:
                ctx.console.print(f"  [red]•[/] {issue.message}")
            return StageResult.failed(
                "Validation failed",
                issues=[str(i.message) for i in result.issues]
            )
        
        return StageResult.ok(
            files_validated=result.files_validated,
            warnings=result.warning_count
        )


class LoadPackageStage(BuildStage):
    name = "load"
    description = "Loading package..."
    weight = StageWeight.LOAD.value
    
    def execute(self, ctx: BuildContext) -> StageResult:
        try:
            ctx.package = COPPackage.load(ctx.package_path)
            
            # Warn about unused config sections
            if ctx.package.manifest:
                warn_dead_config(ctx.package.manifest, verbose=ctx.verbose)
            
            return StageResult.ok(
                package_name=ctx.package.name,
                package_version=ctx.package.version
            )
            
        except FileNotFoundError as e:
            ctx.console.print(f"\n[red]✗ Package not found:[/] {e}")
            return StageResult.failed(f"Package not found: {e}")
        except (yaml.YAMLError, ValueError) as e:
            ctx.console.print(f"\n[red]✗ Invalid package format:[/] {e}")
            return StageResult.failed(f"Invalid package format: {e}")
        except (IOError, OSError) as e:
            ctx.console.print(f"\n[red]✗ Failed to read package:[/] {e}")
            return StageResult.failed(f"Failed to read package: {e}")


class CompileTemplatesStage(BuildStage):
    name = "compile"
    description = "Compiling templates..."
    weight = StageWeight.COMPILE.value
    
    def execute(self, ctx: BuildContext) -> StageResult:
        # Load variables from manifest defaults
        ctx.variables = {}
        if ctx.package.manifest:
            context = ctx.package.manifest.get("context", {})
            system_config = context.get("system", {})
            var_defs = system_config.get("variables", {})
            for var_name, var_config in var_defs.items():
                if isinstance(var_config, dict) and "default" in var_config:
                    ctx.variables[var_name] = var_config["default"]
        
        if ctx.verbose:
            ctx.console.print(f"  [dim]Variables: {ctx.variables}[/]")
        
        ctx.compiled_prompt = ctx.package.compile_prompt(ctx.variables)
        
        return StageResult.ok(
            prompt_length=len(ctx.compiled_prompt),
            variables_used=list(ctx.package.system_prompt.variables) if ctx.package.system_prompt else []
        )


class LinkingStage(BuildStage):
    name = "linking"
    description = "Linking data (embeddings)..."
    weight = StageWeight.LINKING.value
    
    def should_skip(self, ctx: BuildContext) -> Optional[str]:
        linking_config = ctx.build_config.get("linking", {})
        linking_enabled = ctx.enable_linking or linking_config.get("enabled", False)
        if not linking_enabled:
            ctx.set_stage_result(self.name, StageStatus.DISABLED)
            return "Linking not enabled"
        return None
    
    def execute(self, ctx: BuildContext) -> StageResult:
        try:
            from cop.pipeline.linker import DataLinker, LinkingConfig
            
            # Create linking config from manifest
            link_config = LinkingConfig.from_manifest(ctx.build_config)
            
            # Create and store linker for reuse
            ctx.linker = DataLinker(
                package_path=ctx.package_path,
                config=link_config,
                console=ctx.console,
                verbose=ctx.verbose
            )
            
            if ctx.verbose:
                ctx.console.print(f"  [dim]Embedding model: {link_config.embedding_model}[/]")
            
            # Run linking pipeline
            ctx.linking_result = ctx.linker.link()
            ctx.vector_index = ctx.linking_result.index
            
            if ctx.verbose:
                ctx.console.print(
                    f"  [dim]Linked {ctx.linking_result.total_files} files → "
                    f"{ctx.linking_result.total_chunks} chunks[/]"
                )
            
            return StageResult.ok(
                total_files=ctx.linking_result.total_files,
                total_chunks=ctx.linking_result.total_chunks,
                embedding_dimensions=ctx.linking_result.embeddings.dimensions if ctx.linking_result.embeddings else 0
            )
            
        except ImportError as e:
            if ctx.verbose:
                ctx.console.print(f"  Linking skipped: {e}", style="yellow", markup=False)
            return StageResult.skipped(f"Missing dependency: {e}")
        except (APIError, APIConnectionError) as e:
            if ctx.verbose:
                ctx.console.print(f"  Linking failed (API): {e}", style="yellow", markup=False)
            return StageResult.failed(f"Embedding API error: {e}")
        except (IOError, OSError) as e:
            if ctx.verbose:
                ctx.console.print(f"  Linking failed (IO): {e}", style="yellow", markup=False)
            return StageResult.failed(f"File error: {e}")


class SyntheticGenerationStage(BuildStage):
    name = "synthetic"
    description = "Generating synthetic data..."
    weight = StageWeight.SYNTHETIC.value
    
    def should_skip(self, ctx: BuildContext) -> Optional[str]:
        if ctx.skip_synthetic:
            ctx.synthetic_data = []
            return "Synthetic generation skipped by user"
        return None
    
    def execute(self, ctx: BuildContext) -> StageResult:
        # Get synthetic generation config
        synthetic_config = ctx.build_config.get("synthetic", {})
        num_samples = synthetic_config.get("samples", 10)
        temperature = synthetic_config.get("temperature", 0.8)
        scenarios = synthetic_config.get("scenarios", [])
        rag_top_k = synthetic_config.get("rag_top_k", 5)
        
        # RAG is automatically enabled when linking produces a vector index
        use_rag_config = synthetic_config.get("use_rag", None)
        if use_rag_config is None:
            use_rag = ctx.vector_index is not None
        else:
            use_rag = use_rag_config and ctx.vector_index is not None
        
        ctx.synthetic_data = []
        
        try:
            from cop.pipeline.synthetic import SyntheticDataPipeline
            
            pipeline = SyntheticDataPipeline(
                endpoint=ctx.lm_studio_url,
                model=ctx.local_model,
                temperature=temperature,
                num_samples=num_samples,
                console=ctx.console,
                verbose=ctx.verbose,
                vector_index=ctx.vector_index,
                use_rag=use_rag,
                rag_top_k=rag_top_k
            )
            
            if ctx.verbose:
                ctx.console.print(f"  [dim]Endpoint: {ctx.lm_studio_url}[/]")
                ctx.console.print(f"  [dim]Samples: {num_samples}[/]")
                if use_rag:
                    ctx.console.print(f"  [dim]RAG: enabled (top_k={rag_top_k})[/]")
                else:
                    ctx.console.print(f"  [dim]RAG: disabled[/]")
            
            ctx.synthetic_data = pipeline.generate(
                system_prompt=ctx.compiled_prompt,
                personas={name: p.data for name, p in ctx.package.personas.items()},
                guardrails=[{"name": g.name, "rules": g.data} for g in ctx.package.guardrails],
                tools=[{"name": t.name, "definition": t.data} for t in ctx.package.tools],
                scenarios=scenarios if scenarios else None
            )
            
            return StageResult.ok(
                lm_studio_url=ctx.lm_studio_url,
                samples_generated=len(ctx.synthetic_data)
            )
            
        except ImportError as e:
            ctx.add_warning(f"Synthetic generation skipped: {e}")
            return StageResult.skipped(f"Missing dependency: {e}")
        except (APIError, APIConnectionError, RateLimitError) as e:
            ctx.add_warning(f"Synthetic generation failed (API): {e}")
            return StageResult.skipped(f"LLM API error: {e}")
        except (json.JSONDecodeError, ValueError) as e:
            ctx.add_warning(f"Synthetic generation failed (data): {e}")
            return StageResult.skipped(f"Data error: {e}")


class DatasetGenerationStage(BuildStage):
    name = "dataset"
    description = "Generating dataset..."
    weight = StageWeight.DATASET.value
    
    def execute(self, ctx: BuildContext) -> StageResult:
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset directory
        dataset_dir = ctx.output_dir / "data"
        dataset_dir.mkdir(exist_ok=True)
        
        # Build context bundle (stored for reuse)
        ctx.context_bundle = {
            "name": ctx.package.name,
            "version": ctx.package.version,
            "system_prompt": ctx.compiled_prompt,
            "personas": {
                name: p.data for name, p in ctx.package.personas.items()
            },
            "guardrails": [
                {"name": g.name, "priority": g.priority, "rules": g.data}
                for g in ctx.package.guardrails
            ],
            "tools": [
                {"name": t.name, "definition": t.data}
                for t in ctx.package.tools
            ],
        }
        
        # Write dataset JSONL
        ctx.dataset_path = ctx.output_dir / f"{ctx.package.name}-dataset.jsonl"
        with open(ctx.dataset_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "_type": "metadata",
                "name": ctx.package.name,
                "version": ctx.package.version,
                "system_prompt": ctx.compiled_prompt
            }) + "\n")
            
            if ctx.synthetic_data:
                for example in ctx.synthetic_data:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        return StageResult.ok(
            dataset_path=str(ctx.dataset_path),
            examples=len(ctx.synthetic_data) if ctx.synthetic_data else 0
        )


class SaveEmbeddingsStage(BuildStage):
    name = "embeddings"
    description = "Saving embeddings index..."
    weight = StageWeight.EMBEDDINGS.value
    
    def should_skip(self, ctx: BuildContext) -> Optional[str]:
        if ctx.linking_result is None:
            return "No linking result to save"
        return None
    
    def execute(self, ctx: BuildContext) -> StageResult:
        try:
            # Reuse the linker from linking stage
            if ctx.linker is None:
                from cop.pipeline.linker import DataLinker
                ctx.linker = DataLinker(
                    package_path=ctx.package_path,
                    console=ctx.console,
                    verbose=ctx.verbose
                )
            
            index_dir = ctx.output_dir / "embeddings"
            ctx.linker.save_artifacts(ctx.linking_result, index_dir)
            
            if ctx.verbose:
                ctx.console.print(f"  [dim]Saved embeddings to {index_dir}[/]")
            
            return StageResult.ok(
                index_dir=str(index_dir),
                chunks=ctx.linking_result.total_chunks,
                dimensions=ctx.linking_result.embeddings.dimensions if ctx.linking_result.embeddings else 0
            )
            
        except (IOError, OSError) as e:
            if ctx.verbose:
                ctx.console.print(f"  Failed to save embeddings: {e}", style="yellow", markup=False)
            return StageResult.failed(f"Failed to write embeddings: {e}")
        except (json.JSONDecodeError, TypeError) as e:
            if ctx.verbose:
                ctx.console.print(f"  Failed to serialize embeddings: {e}", style="yellow", markup=False)
            return StageResult.failed(f"Serialization error: {e}")


class CreateConfigStage(BuildStage):
    name = "config"
    description = "Creating config..."
    weight = StageWeight.CONFIG.value
    
    def execute(self, ctx: BuildContext) -> StageResult:
        config = {
            "base_model": "meta-llama/Llama-3.2-3B-Instruct",
            "package": {
                "name": ctx.package.name,
                "version": ctx.package.version
            },
            "training": {
                "method": "lora",
                "lora_r": 16,
                "lora_alpha": 32,
                "epochs": 3
            },
            "output": {
                "export_gguf": True,
                "quantization": "Q4_K_M"
            }
        }
        
        config_path = ctx.output_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return StageResult.ok()


class CreateManifestStage(BuildStage):
    name = "manifest"
    description = "Creating manifest..."
    weight = StageWeight.MANIFEST.value
    
    def execute(self, ctx: BuildContext) -> StageResult:
        manifest = {
            "version": "1.0.0",
            "name": ctx.package.name,
            "package_version": ctx.package.version,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "build_state": ctx.build_state,
            "files": {}
        }
        
        # Calculate checksums
        for file_path in ctx.output_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "manifest.json":
                with open(file_path, "rb") as f:
                    checksum = hashlib.sha256(f.read()).hexdigest()
                manifest["files"][str(file_path.relative_to(ctx.output_dir))] = {
                    "checksum": f"sha256:{checksum}",
                    "size": file_path.stat().st_size
                }
        
        manifest_path = ctx.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        
        return StageResult.ok()


class PackageArtifactStage(BuildStage):
    name = "package"
    description = "Packaging artifact..."
    weight = StageWeight.PACKAGE.value
    
    def execute(self, ctx: BuildContext) -> StageResult:
        artifact_name = f"{ctx.package.name}-{ctx.package.version}.ftpack"
        ctx.artifact_path = ctx.output_dir / artifact_name
        
        with tarfile.open(ctx.artifact_path, "w:gz") as tar:
            for file_path in ctx.output_dir.rglob("*"):
                if file_path.is_file() and not file_path.name.endswith(".ftpack"):
                    tar.add(file_path, arcname=file_path.relative_to(ctx.output_dir))
        
        return StageResult.ok(
            artifact=str(ctx.artifact_path),
            size=ctx.artifact_path.stat().st_size
        )


# =============================================================================
# Build Pipeline
# =============================================================================

class BuildPipeline:
    """Orchestrates the build process through discrete stages."""
    
    def __init__(self, ctx: BuildContext):
        self.ctx = ctx
        self.stages: list[BuildStage] = []
    
    def add_stage(self, stage: BuildStage) -> BuildPipeline:
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        return self
    
    def run(self) -> bool:
        """Execute all stages in sequence."""
        for stage in self.stages:
            result = stage.run(self.ctx)
            if not result.success:
                return False
        return True


def _write_context_bundle(ctx: BuildContext) -> None:
    """Write the context bundle to disk. Extracted to avoid duplication."""
    context_path = ctx.output_dir / "context.bundle.json"
    with open(context_path, "w", encoding="utf-8") as f:
        json.dump(ctx.context_bundle, f, indent=2)


def _print_summary(ctx: BuildContext, dataset_only: bool = False) -> None:
    """Print build summary."""
    ctx.console.print()
    
    if dataset_only:
        ctx.console.print("[bold green]✓ Dataset generated successfully![/]")
        ctx.console.print()
        ctx.console.print(f"  [bold]Dataset:[/] {ctx.dataset_path}")
        ctx.console.print(f"  [bold]Context:[/] {ctx.output_dir / 'context.bundle.json'}")
    else:
        ctx.console.print("[bold green]✓ Build completed successfully![/]")
        ctx.console.print()
        ctx.console.print(f"  [bold]Artifact:[/] {ctx.artifact_path}")
        ctx.console.print(f"  [bold]Size:[/] {ctx.artifact_path.stat().st_size / 1024:.1f} KB")
    
    ctx.console.print()
    
    if ctx.build_state["warnings"]:
        ctx.console.print("[yellow]Warnings:[/]")
        for warning in ctx.build_state["warnings"]:
            ctx.console.print(f"  • {warning}")
        ctx.console.print()


# =============================================================================
# Main Entry Point
# =============================================================================

def run_build(
    package_path: Path,
    output_dir: Path,
    skip_validation: bool = False,
    skip_synthetic: bool = False,
    lm_studio_url: Optional[str] = None,
    dataset_only: bool = False,
    enable_linking: bool = False,
    verbose: bool = False,
    console: Optional[Console] = None
) -> bool:
    """Build a COP package into deployable artifacts.
    
    Args:
        package_path: Path to the COP package directory
        output_dir: Output directory for artifacts
        skip_validation: Skip validation step
        skip_synthetic: Skip synthetic data generation
        lm_studio_url: LM Studio API URL (overrides config)
        dataset_only: Output only JSONL dataset, skip packaging
        enable_linking: Enable data linking (embeddings/RAG)
        verbose: Show detailed output
        console: Rich console for output
    
    Returns:
        True if build succeeded, False otherwise
    """
    console = console or Console()
    
    # Load cop.yaml to get build config
    cop_yaml_path = package_path / "cop.yaml"
    build_config = {}
    if cop_yaml_path.exists():
        with open(cop_yaml_path, encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
            build_config = manifest.get("build", {})
    
    # Get LM Studio URL from: CLI arg > config > default
    local_llm_config = build_config.get("local_llm", {})
    if lm_studio_url is None:
        lm_studio_url = local_llm_config.get("endpoint", DEFAULT_LLM_ENDPOINT)
    
    local_model = local_llm_config.get("model", DEFAULT_LLM_MODEL)
    
    # Display build header
    console.print()
    console.print(Panel(
        f"[bold]Building:[/] {package_path}\n[bold]Output:[/] {output_dir}",
        title="[cyan]COP Build[/]",
        border_style="cyan"
    ))
    console.print()
    
    # Initialize build context
    ctx = BuildContext(
        package_path=package_path,
        output_dir=Path(output_dir),
        skip_validation=skip_validation,
        skip_synthetic=skip_synthetic,
        dataset_only=dataset_only,
        enable_linking=enable_linking,
        verbose=verbose,
        lm_studio_url=lm_studio_url,
        local_model=local_model,
        build_config=build_config,
        console=console,
    )
    
    ctx.build_state.update({
        "package_path": str(package_path),
        "output_dir": str(output_dir),
        "started_at": datetime.utcnow().isoformat(),
    })
    
    # Run build with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        ctx.progress = progress
        ctx.progress_task = progress.add_task("[cyan]Building package...", total=100)
        
        # Build the pipeline
        pipeline = BuildPipeline(ctx)
        pipeline.add_stage(ValidationStage())
        pipeline.add_stage(LoadPackageStage())
        pipeline.add_stage(CompileTemplatesStage())
        pipeline.add_stage(LinkingStage())
        pipeline.add_stage(SyntheticGenerationStage())
        pipeline.add_stage(DatasetGenerationStage())
        
        # Run common stages
        if not pipeline.run():
            return False
        
        # Handle dataset_only mode
        if dataset_only:
            _write_context_bundle(ctx)
            ctx.update_progress(
                advance=StageWeight.DATASET_ONLY_REMAINING.value,
                description="[green]Dataset complete!"
            )
            _print_summary(ctx, dataset_only=True)
            return True
        
        # Full build: continue with remaining stages
        _write_context_bundle(ctx)
        
        full_build_pipeline = BuildPipeline(ctx)
        full_build_pipeline.add_stage(SaveEmbeddingsStage())
        full_build_pipeline.add_stage(CreateConfigStage())
        full_build_pipeline.add_stage(CreateManifestStage())
        full_build_pipeline.add_stage(PackageArtifactStage())
        
        if not full_build_pipeline.run():
            return False
        
        ctx.update_progress(description="[green]Build complete!")
    
    _print_summary(ctx)
    return True
