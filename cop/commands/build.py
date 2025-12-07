"""
COP Build Command - Build COP packages into deployable artifacts
"""

import hashlib
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from openai import APIError, APIConnectionError, RateLimitError
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from cop.core.package import COPPackage
from cop.core.validator import COPValidator
from cop.pipeline.constants import DEFAULT_LLM_ENDPOINT, DEFAULT_LLM_MODEL, warn_dead_config


def run_build(
    package_path: Path,
    output_dir: Path,
    skip_validation: bool = False,
    skip_synthetic: bool = False,
    lm_studio_url: str = None,  # None means use config or default
    dataset_only: bool = False,
    enable_linking: bool = False,  # Enable data linking (embeddings/RAG)
    verbose: bool = False,
    console: Console = None
) -> bool:
    """Build a COP package into deployable artifacts.
    
    Args:
        package_path: Path to the COP package directory
        output_dir: Output directory for artifacts
        skip_validation: Skip validation step
        skip_synthetic: Skip synthetic data generation
        lm_studio_url: LM Studio API URL (overrides config)
        dataset_only: Output only JSONL dataset, skip packaging
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
        import yaml
        with open(cop_yaml_path, encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
            build_config = manifest.get("build", {})
    
    # Get LM Studio URL from: CLI arg > config > env var > default
    local_llm_config = build_config.get("local_llm", {})
    if lm_studio_url is None:
        lm_studio_url = local_llm_config.get("endpoint", DEFAULT_LLM_ENDPOINT)
    
    # Get model name from config
    local_model = local_llm_config.get("model", DEFAULT_LLM_MODEL)
    
    console.print()
    console.print(Panel(
        f"[bold]Building:[/] {package_path}\n[bold]Output:[/] {output_dir}",
        title="[cyan]COP Build[/]",
        border_style="cyan"
    ))
    console.print()
    
    build_state = {
        "package_path": str(package_path),
        "output_dir": str(output_dir),
        "started_at": datetime.utcnow().isoformat(),
        "stages": {},
        "errors": [],
        "warnings": [],
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        overall = progress.add_task("[cyan]Building package...", total=100)
        
        # Stage 1: Validation
        if not skip_validation:
            progress.update(overall, description="[cyan]Validating package...")
            
            validator = COPValidator()
            result = validator.validate_package(package_path)
            
            if result.has_errors:
                console.print()
                console.print("[red]✗ Validation failed[/]")
                for issue in result.issues:
                    console.print(f"  [red]•[/] {issue.message}")
                return False
            
            build_state["stages"]["validation"] = {
                "status": "passed",
                "files_validated": result.files_validated,
                "warnings": result.warning_count
            }
            
            progress.update(overall, advance=15)
        else:
            progress.update(overall, advance=15)
            build_state["stages"]["validation"] = {"status": "skipped"}
        
        # Stage 2: Load package
        progress.update(overall, description="[cyan]Loading package...")
        
        try:
            package = COPPackage.load(package_path)
            build_state["stages"]["load"] = {
                "status": "success",
                "name": package.name,
                "version": package.version
            }
            
            # Warn about unused config sections
            if package.manifest:
                warn_dead_config(package.manifest, verbose=verbose)
                
        except FileNotFoundError as e:
            console.print(f"\n[red]✗ Package not found:[/] {e}")
            return False
        except (yaml.YAMLError, ValueError) as e:
            console.print(f"\n[red]✗ Invalid package format:[/] {e}")
            return False
        except (IOError, OSError) as e:
            console.print(f"\n[red]✗ Failed to read package:[/] {e}")
            return False
        
        progress.update(overall, advance=15)
        
        # Stage 3: Compile templates
        progress.update(overall, description="[cyan]Compiling templates...")
        
        # Load variables from manifest defaults
        variables = {}
        if package.manifest:
            context = package.manifest.get("context", {})
            system_config = context.get("system", {})
            var_defs = system_config.get("variables", {})
            for var_name, var_config in var_defs.items():
                if isinstance(var_config, dict) and "default" in var_config:
                    variables[var_name] = var_config["default"]
        
        if verbose:
            console.print(f"  [dim]Variables: {variables}[/]")
        
        compiled_prompt = package.compile_prompt(variables)
        
        build_state["stages"]["compile"] = {
            "status": "success",
            "prompt_length": len(compiled_prompt),
            "variables_used": list(package.system_prompt.variables) if package.system_prompt else []
        }
        
        progress.update(overall, advance=15)
        
        # Stage 3.5: Data Linking (optional)
        linking_result = None
        vector_index = None
        
        linking_config = build_config.get("linking", {})
        linking_enabled = enable_linking or linking_config.get("enabled", False)
        
        if linking_enabled:
            progress.update(overall, description="[cyan]Linking data (embeddings)...")
            
            try:
                from cop.pipeline.linker import DataLinker, LinkingConfig
                
                # Create linking config from manifest
                link_config = LinkingConfig.from_manifest(build_config)
                
                linker = DataLinker(
                    package_path=package_path,
                    config=link_config,
                    console=console,
                    verbose=verbose
                )
                
                if verbose:
                    console.print(f"  [dim]Embedding model: {link_config.embedding_model}[/]")
                
                # Run linking pipeline
                linking_result = linker.link()
                vector_index = linking_result.index
                
                build_state["stages"]["linking"] = {
                    "status": "success",
                    "total_files": linking_result.total_files,
                    "total_chunks": linking_result.total_chunks,
                    "embedding_dimensions": linking_result.embeddings.dimensions if linking_result.embeddings else 0,
                }
                
                if verbose:
                    console.print(f"  [dim]Linked {linking_result.total_files} files → {linking_result.total_chunks} chunks[/]")
                
            except ImportError as e:
                build_state["stages"]["linking"] = {
                    "status": "skipped",
                    "reason": f"Missing dependency: {e}"
                }
                if verbose:
                    console.print(f"  Linking skipped: {e}", style="yellow", markup=False)
            except (APIError, APIConnectionError) as e:
                build_state["stages"]["linking"] = {
                    "status": "failed",
                    "reason": f"Embedding API error: {e}"
                }
                if verbose:
                    console.print(f"  Linking failed (API): {e}", style="yellow", markup=False)
                return False
            except (IOError, OSError) as e:
                build_state["stages"]["linking"] = {
                    "status": "failed",
                    "reason": f"File error: {e}"
                }
                if verbose:
                    console.print(f"  Linking failed (IO): {e}", style="yellow", markup=False)
                return False
        else:
            build_state["stages"]["linking"] = {"status": "disabled"}
        
        # Stage 4: Synthetic generation (optional)
        if not skip_synthetic:
            progress.update(overall, description="[cyan]Generating synthetic data...")
            
            # Get synthetic generation config
            synthetic_config = build_config.get("synthetic", {})
            num_samples = synthetic_config.get("samples", 10)
            temperature = synthetic_config.get("temperature", 0.8)
            scenarios = synthetic_config.get("scenarios", [])
            rag_top_k = synthetic_config.get("rag_top_k", 5)
            
            # RAG is automatically enabled when linking produces a vector index
            # Can be explicitly disabled with use_rag: false in config
            use_rag_config = synthetic_config.get("use_rag", None)
            if use_rag_config is None:
                # Auto-enable RAG when vector_index exists
                use_rag = vector_index is not None
            else:
                # Explicit config overrides auto-detection
                use_rag = use_rag_config and vector_index is not None
            
            synthetic_data = []
            try:
                from cop.pipeline.synthetic import SyntheticDataPipeline
                
                # Create pipeline with RAG (auto-enabled when linking is used)
                pipeline = SyntheticDataPipeline(
                    endpoint=lm_studio_url,
                    model=local_model,
                    temperature=temperature,
                    num_samples=num_samples,
                    console=console,
                    verbose=verbose,
                    vector_index=vector_index,
                    use_rag=use_rag,
                    rag_top_k=rag_top_k
                )
                
                if verbose:
                    console.print(f"  [dim]Endpoint: {lm_studio_url}[/]")
                    console.print(f"  [dim]Samples: {num_samples}[/]")
                    if use_rag:
                        console.print(f"  [dim]RAG: enabled (top_k={rag_top_k})[/]")
                    else:
                        console.print(f"  [dim]RAG: disabled[/]")
                
                # Generate synthetic data
                synthetic_data = pipeline.generate(
                    system_prompt=compiled_prompt,
                    personas={name: p.data for name, p in package.personas.items()},
                    guardrails=[{"name": g.name, "rules": g.data} for g in package.guardrails],
                    tools=[{"name": t.name, "definition": t.data} for t in package.tools],
                    scenarios=scenarios if scenarios else None
                )
                
                build_state["stages"]["synthetic"] = {
                    "status": "success",
                    "lm_studio_url": lm_studio_url,
                    "samples_generated": len(synthetic_data),
                }
                
            except ImportError as e:
                build_state["stages"]["synthetic"] = {
                    "status": "skipped",
                    "reason": f"Missing dependency: {e}"
                }
                build_state["warnings"].append(f"Synthetic generation skipped: {e}")
            except (APIError, APIConnectionError, RateLimitError) as e:
                build_state["stages"]["synthetic"] = {
                    "status": "skipped",
                    "reason": f"LLM API error: {e}"
                }
                build_state["warnings"].append(f"Synthetic generation failed (API): {e}")
            except (json.JSONDecodeError, ValueError) as e:
                build_state["stages"]["synthetic"] = {
                    "status": "skipped",
                    "reason": f"Data error: {e}"
                }
                build_state["warnings"].append(f"Synthetic generation failed (data): {e}")
            
            progress.update(overall, advance=15)
        else:
            progress.update(overall, advance=15)
            build_state["stages"]["synthetic"] = {"status": "skipped"}
            synthetic_data = []
        
        # Stage 5: Generate dataset
        progress.update(overall, description="[cyan]Generating dataset...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset directory
        dataset_dir = output_dir / "data"
        dataset_dir.mkdir(exist_ok=True)
        
        # Write compiled context
        context_bundle = {
            "name": package.name,
            "version": package.version,
            "system_prompt": compiled_prompt,
            "personas": {
                name: p.data for name, p in package.personas.items()
            },
            "guardrails": [
                {"name": g.name, "priority": g.priority, "rules": g.data}
                for g in package.guardrails
            ],
            "tools": [
                {"name": t.name, "definition": t.data}
                for t in package.tools
            ],
        }
        
        # Write dataset JSONL (training format)
        dataset_path = output_dir / f"{package.name}-dataset.jsonl"
        with open(dataset_path, "w", encoding="utf-8") as f:
            # Write context as first entry (metadata)
            f.write(json.dumps({
                "_type": "metadata",
                "name": package.name,
                "version": package.version,
                "system_prompt": compiled_prompt
            }) + "\n")
            
            # Write synthetic training examples
            if synthetic_data:
                for example in synthetic_data:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        build_state["stages"]["dataset"] = {
            "status": "success",
            "dataset_path": str(dataset_path),
            "examples": len(synthetic_data) if synthetic_data else 0
        }
        
        # If dataset_only, stop here
        if dataset_only:
            # Also write context bundle for reference
            context_path = output_dir / "context.bundle.json"
            with open(context_path, "w", encoding="utf-8") as f:
                json.dump(context_bundle, f, indent=2)
            
            progress.update(overall, advance=40, description="[green]Dataset complete!")
            
            # Summary for dataset-only mode
            console.print()
            console.print("[bold green]✓ Dataset generated successfully![/]")
            console.print()
            console.print(f"  [bold]Dataset:[/] {dataset_path}")
            console.print(f"  [bold]Context:[/] {context_path}")
            console.print()
            
            if build_state["warnings"]:
                console.print("[yellow]Warnings:[/]")
                for warning in build_state["warnings"]:
                    console.print(f"  • {warning}")
                console.print()
            
            return True
        
        # Full build: write context bundle
        context_path = output_dir / "context.bundle.json"
        with open(context_path, "w", encoding="utf-8") as f:
            json.dump(context_bundle, f, indent=2)
        
        # Save linking artifacts if available
        if linking_result is not None:
            progress.update(overall, description="[cyan]Saving embeddings index...")
            
            try:
                from cop.pipeline.linker import DataLinker
                
                linker = DataLinker(
                    package_path=package_path,
                    console=console,
                    verbose=verbose
                )
                
                # Save index and chunks to output directory
                index_dir = output_dir / "embeddings"
                linker.save_artifacts(linking_result, index_dir)
                
                build_state["stages"]["embeddings"] = {
                    "status": "success",
                    "index_dir": str(index_dir),
                    "chunks": linking_result.total_chunks,
                    "dimensions": linking_result.embeddings.dimensions if linking_result.embeddings else 0
                }
                
                if verbose:
                    console.print(f"  [dim]Saved embeddings to {index_dir}[/]")
                    
            except (IOError, OSError) as e:
                build_state["stages"]["embeddings"] = {
                    "status": "failed",
                    "reason": f"Failed to write embeddings: {e}"
                }
                if verbose:
                    console.print(f"  Failed to save embeddings: {e}", style="yellow", markup=False)
            except (json.JSONDecodeError, TypeError) as e:
                build_state["stages"]["embeddings"] = {
                    "status": "failed",
                    "reason": f"Serialization error: {e}"
                }
                if verbose:
                    console.print(f"  Failed to serialize embeddings: {e}", style="yellow", markup=False)
        
        progress.update(overall, advance=15)
        
        # Stage 6: Create config
        progress.update(overall, description="[cyan]Creating config...")
        
        config = {
            "base_model": "meta-llama/Llama-3.2-3B-Instruct",
            "package": {
                "name": package.name,
                "version": package.version
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
        
        config_path = output_dir / "config.yaml"
        import yaml
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        build_state["stages"]["config"] = {"status": "success"}
        
        progress.update(overall, advance=10)
        
        # Stage 7: Create metadata/manifest
        progress.update(overall, description="[cyan]Creating manifest...")
        
        manifest = {
            "version": "1.0.0",
            "name": package.name,
            "package_version": package.version,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "build_state": build_state,
            "files": {}
        }
        
        # Calculate checksums
        for file_path in output_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "manifest.json":
                with open(file_path, "rb") as f:
                    checksum = hashlib.sha256(f.read()).hexdigest()
                manifest["files"][str(file_path.relative_to(output_dir))] = {
                    "checksum": f"sha256:{checksum}",
                    "size": file_path.stat().st_size
                }
        
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        
        build_state["stages"]["manifest"] = {"status": "success"}
        
        progress.update(overall, advance=10)
        
        # Stage 8: Package artifact
        progress.update(overall, description="[cyan]Packaging artifact...")
        
        artifact_name = f"{package.name}-{package.version}.ftpack"
        artifact_path = output_dir / artifact_name
        
        with tarfile.open(artifact_path, "w:gz") as tar:
            for file_path in output_dir.rglob("*"):
                if file_path.is_file() and not file_path.name.endswith(".ftpack"):
                    tar.add(file_path, arcname=file_path.relative_to(output_dir))
        
        build_state["stages"]["package"] = {
            "status": "success",
            "artifact": str(artifact_path),
            "size": artifact_path.stat().st_size
        }
        
        progress.update(overall, advance=5, description="[green]Build complete!")
    
    # Summary
    console.print()
    console.print("[bold green]✓ Build completed successfully![/]")
    console.print()
    console.print(f"  [bold]Artifact:[/] {artifact_path}")
    console.print(f"  [bold]Size:[/] {artifact_path.stat().st_size / 1024:.1f} KB")
    console.print()
    
    if build_state["warnings"]:
        console.print("[yellow]Warnings:[/]")
        for warning in build_state["warnings"]:
            console.print(f"  • {warning}")
        console.print()
    
    return True

