#!/usr/bin/env python3
"""
COP CLI - Main command entrypoint

Usage:
    cop --help
    cop validate <path>
    cop build <path>
    cop init <name>
    cop serve
    cop info <path>
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Initialize rich console
console = Console()

# ASCII art banner
BANNER = r"""
   ____ ___  ____     ____ _     ___ 
  / ___/ _ \|  _ \   / ___| |   |_ _|
 | |  | | | | |_) | | |   | |    | | 
 | |__| |_| |  __/  | |___| |___ | | 
  \____\___/|_|      \____|_____|___|
                                     
  Context-Oriented Programming CLI
"""


@click.group(invoke_without_command=True)
@click.option("--version", "-V", is_flag=True, help="Show version and exit")
@click.pass_context
def main(ctx, version):
    """COP CLI - Build and deploy LLM agents from context modules.
    
    \b
    Commands:
      validate  Validate a COP package against schemas
      build     Build a COP package into deployable artifacts
      init      Initialize a new COP package
      serve     Start local LLM server integration
      info      Show information about a COP package
    """
    if version:
        from cop import __version__
        console.print(f"cop-cli version {__version__}")
        sys.exit(0)
    
    if ctx.invoked_subcommand is None:
        console.print(Panel(BANNER, title="[bold cyan]COP CLI[/]", border_style="cyan"))
        console.print()
        console.print("Use [bold]cop --help[/] for usage information.")
        console.print()


@main.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--strict", is_flag=True, help="Treat warnings as errors")
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text")
def validate(path, verbose, strict, output_format):
    """Validate a COP package against schemas.
    
    \b
    Examples:
      cop validate .
      cop validate examples/customer-support-agent
      cop validate --verbose --strict .
    """
    from cop.commands.validate import run_validate
    
    result = run_validate(
        package_path=Path(path),
        verbose=verbose,
        strict=strict,
        output_format=output_format,
        console=console
    )
    
    sys.exit(0 if result else 1)


@main.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--output", "-o", type=click.Path(), default="dist", help="Output directory")
@click.option("--target", "-t", multiple=True, 
              type=click.Choice(["openai", "anthropic", "local", "all"]),
              default=["all"], help="Target format(s)")
@click.option("--skip-validation", is_flag=True, help="Skip validation step")
@click.option("--skip-synthetic", is_flag=True, help="Skip synthetic data generation")
@click.option("--lm-studio-url", default=None, 
              help="LM Studio API URL (overrides cop.yaml config)")
@click.option("--dataset-only", is_flag=True, help="Output only JSONL dataset, skip packaging")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def build(path, output, target, skip_validation, skip_synthetic, lm_studio_url, dataset_only, verbose):
    """Build a COP package into deployable artifacts.
    
    \b
    This command orchestrates the full build pipeline:
      1. Load and parse COP manifest
      2. Validate package structure
      3. Expand template variables
      4. Generate synthetic data (optional)
      5. Create training datasets
      6. Package into .ftpack artifact
    
    \b
    Examples:
      cop build .
      cop build examples/customer-support-agent -o ./output
      cop build . --target local --skip-synthetic
    """
    from cop.commands.build import run_build
    
    result = run_build(
        package_path=Path(path),
        output_dir=Path(output),
        targets=list(target),
        skip_validation=skip_validation,
        skip_synthetic=skip_synthetic,
        lm_studio_url=lm_studio_url,
        dataset_only=dataset_only,
        verbose=verbose,
        console=console
    )
    
    sys.exit(0 if result else 1)


@main.command()
@click.argument("name")
@click.option("--template", "-t", 
              type=click.Choice(["basic", "customer-support", "coding-assistant"]),
              default="basic", help="Package template to use")
@click.option("--output", "-o", type=click.Path(), default=".", help="Output directory")
def init(name, template, output):
    """Initialize a new COP package.
    
    \b
    Creates a new COP package with the following structure:
      <name>/
        cop.yaml          - Package manifest
        prompts/
          system.md       - System prompt
        personas/
          default.yaml    - Default persona
        guardrails/
          safety.yaml     - Safety guardrails
        knowledge/
        tools/
        tests/
    
    \b
    Examples:
      cop init my-agent
      cop init customer-bot --template customer-support
      cop init assistant -o ./projects
    """
    from cop.commands.init import run_init
    
    result = run_init(
        name=name,
        template=template,
        output_dir=Path(output),
        console=console
    )
    
    sys.exit(0 if result else 1)


@main.command()
@click.option("--port", "-p", default=8080, help="Server port")
@click.option("--lm-studio-url", default="http://localhost:1234/v1",
              help="LM Studio API URL")
@click.option("--package", type=click.Path(exists=True), help="COP package to serve")
def serve(port, lm_studio_url, package):
    """Start local development server with LM Studio integration.
    
    \b
    This command starts a local server that:
      - Loads a COP package context
      - Proxies requests to LM Studio
      - Applies persona, guardrails, and context
    
    \b
    Prerequisites:
      1. LM Studio must be running with a model loaded
      2. LM Studio server must be started (localhost:1234)
    
    \b
    Examples:
      cop serve
      cop serve --port 8000
      cop serve --package ./my-agent
    """
    from cop.commands.serve import run_serve
    
    run_serve(
        port=port,
        lm_studio_url=lm_studio_url,
        package_path=Path(package) if package else None,
        console=console
    )


@main.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--format", "output_format", type=click.Choice(["text", "json", "yaml"]),
              default="text")
def info(path, output_format):
    """Show information about a COP package.
    
    \b
    Displays:
      - Package metadata (name, version, description)
      - Context modules (prompts, personas, guardrails)
      - Tools and knowledge sources
      - Test suites
    
    \b
    Examples:
      cop info .
      cop info examples/customer-support-agent
      cop info . --format json
    """
    from cop.commands.info import run_info
    
    result = run_info(
        package_path=Path(path),
        output_format=output_format,
        console=console
    )
    
    sys.exit(0 if result else 1)


@main.command()
@click.argument("ftpack", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--list", "-l", "list_only", is_flag=True, help="List contents only")
def extract(ftpack, output, list_only):
    """Extract a .ftpack artifact.
    
    \b
    Examples:
      cop extract dist/my-agent-1.0.0.ftpack
      cop extract my-agent.ftpack -o ./extracted
      cop extract my-agent.ftpack --list
    """
    from cop.commands.extract import run_extract
    
    result = run_extract(
        ftpack_path=Path(ftpack),
        output_dir=Path(output) if output else None,
        list_only=list_only,
        console=console
    )
    sys.exit(0 if result else 1)


@main.command()
def doctor():
    """Check system dependencies and configuration.
    
    \b
    Checks:
      - Python version
      - Required packages
      - LM Studio connectivity
      - Schema files
    """
    from cop.commands.doctor import run_doctor
    
    result = run_doctor(console=console)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()

