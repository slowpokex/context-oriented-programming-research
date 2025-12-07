"""
COP Build Command - Build COP packages into deployable artifacts
"""

import hashlib
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from cop.core.package import COPPackage
from cop.core.validator import COPValidator


def run_build(
    package_path: Path,
    output_dir: Path,
    targets: List[str],
    skip_validation: bool = False,
    skip_synthetic: bool = False,
    lm_studio_url: str = None,  # None means use config or default
    dataset_only: bool = False,
    verbose: bool = False,
    console: Console = None
) -> bool:
    """Build a COP package into deployable artifacts.
    
    Args:
        package_path: Path to the COP package directory
        output_dir: Output directory for artifacts
        targets: Target formats to generate
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
    
    # Get LM Studio URL from: CLI arg > config > default
    local_llm_config = build_config.get("local_llm", {})
    if lm_studio_url is None:
        lm_studio_url = local_llm_config.get("endpoint", "http://localhost:1234/v1")
    
    # Get model name from config
    local_model = local_llm_config.get("model", "local-model")
    
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
        except Exception as e:
            console.print(f"\n[red]✗ Failed to load package:[/] {e}")
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
        
        # Stage 4: Synthetic generation (optional)
        if not skip_synthetic:
            progress.update(overall, description="[cyan]Generating synthetic data...")
            
            # Get synthetic generation config
            synthetic_config = build_config.get("synthetic", {})
            num_samples = synthetic_config.get("samples", 10)
            temperature = synthetic_config.get("temperature", 0.8)
            scenarios = synthetic_config.get("scenarios", [])
            
            synthetic_data = []
            try:
                from cop.pipeline.synthetic import SyntheticDataPipeline
                
                # Create pipeline
                pipeline = SyntheticDataPipeline(
                    endpoint=lm_studio_url,
                    model=local_model,
                    temperature=temperature,
                    num_samples=num_samples,
                    console=console,
                    verbose=verbose
                )
                
                if verbose:
                    console.print(f"  [dim]Endpoint: {lm_studio_url}[/]")
                    console.print(f"  [dim]Samples: {num_samples}[/]")
                
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
            except Exception as e:
                build_state["stages"]["synthetic"] = {
                    "status": "skipped",
                    "reason": f"Error: {e}"
                }
                build_state["warnings"].append(f"Synthetic generation failed: {e}")
            
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

