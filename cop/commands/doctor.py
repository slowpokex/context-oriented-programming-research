"""
COP Doctor Command - Check system dependencies and configuration
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def run_doctor(console: Console = None) -> bool:
    """Check system dependencies and configuration.
    
    Args:
        console: Rich console for output
    
    Returns:
        True if all checks pass, False otherwise
    """
    console = console or Console()
    
    console.print()
    console.print(Panel(
        "[bold]System Diagnostics[/]",
        title="[cyan]COP Doctor[/]",
        border_style="cyan"
    ))
    console.print()
    
    all_passed = True
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Details", style="dim")
    
    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 10)
    table.add_row(
        "Python version",
        "[green]✓[/]" if py_ok else "[red]✗[/]",
        f"{py_version} {'(OK)' if py_ok else '(need >=3.10)'}"
    )
    if not py_ok:
        all_passed = False
    
    # Required packages
    packages = [
        ("click", "CLI framework"),
        ("pyyaml", "YAML parsing"),
        ("jsonschema", "Schema validation"),
        ("rich", "Terminal output"),
    ]
    
    for pkg_name, description in packages:
        try:
            pkg = __import__(pkg_name.replace("-", "_"))
            version = getattr(pkg, "__version__", "installed")
            table.add_row(pkg_name, "[green]✓[/]", f"{version} - {description}")
        except ImportError:
            table.add_row(pkg_name, "[red]✗[/]", f"Not installed - {description}")
            all_passed = False
    
    # Optional packages
    optional_packages = [
        ("openai", "LM Studio integration"),
        ("langgraph", "Pipeline orchestration"),
        ("diskcache", "Response caching"),
    ]
    
    for pkg_name, description in optional_packages:
        try:
            pkg = __import__(pkg_name.replace("-", "_"))
            version = getattr(pkg, "__version__", "installed")
            table.add_row(f"{pkg_name} (optional)", "[green]✓[/]", f"{version} - {description}")
        except ImportError:
            table.add_row(f"{pkg_name} (optional)", "[yellow]○[/]", f"Not installed - {description}")
    
    # Schema files
    schema_dir = Path(__file__).parent.parent.parent / "schemas"
    schemas = ["cop-manifest", "persona", "guardrail", "tool", "test"]
    
    for schema_name in schemas:
        schema_path = schema_dir / f"{schema_name}.schema.json"
        if schema_path.exists():
            table.add_row(f"{schema_name} schema", "[green]✓[/]", str(schema_path))
        else:
            table.add_row(f"{schema_name} schema", "[red]✗[/]", "Not found")
            all_passed = False
    
    # LM Studio connectivity
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
        models = client.models.list()
        if models.data:
            model_names = ", ".join(m.id for m in models.data[:2])
            table.add_row("LM Studio", "[green]✓[/]", f"Connected ({model_names}...)")
        else:
            table.add_row("LM Studio", "[yellow]○[/]", "Connected but no models loaded")
    except ImportError:
        table.add_row("LM Studio", "[yellow]○[/]", "openai package not installed")
    except Exception:
        table.add_row("LM Studio", "[yellow]○[/]", "Not running or not accessible")
    
    console.print(table)
    console.print()
    
    if all_passed:
        console.print("[bold green]✓ All required checks passed![/]")
    else:
        console.print("[bold red]✗ Some checks failed[/]")
        console.print()
        console.print("To install missing packages:")
        console.print("  [cyan]pip install cop-cli[/]")
        console.print("  or")
        console.print("  [cyan]pip install -e .[/] (for development)")
    
    console.print()
    return all_passed

