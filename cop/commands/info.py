"""
COP Info Command - Display information about a COP package
"""

import json
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from cop.core.package import COPPackage


def run_info(
    package_path: Path,
    output_format: str = "text",
    console: Console = None
) -> bool:
    """Display information about a COP package.
    
    Args:
        package_path: Path to the COP package directory
        output_format: Output format (text, json, or yaml)
        console: Rich console for output
    
    Returns:
        True if successful, False otherwise
    """
    console = console or Console()
    
    try:
        package = COPPackage.load(package_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/] {e}")
        return False
    except Exception as e:
        console.print(f"[red]Error loading package:[/] {e}")
        return False
    
    if output_format == "json":
        console.print_json(data=package.to_dict())
        return True
    
    if output_format == "yaml":
        console.print(yaml.dump(package.to_dict(), default_flow_style=False))
        return True
    
    # Text output
    console.print()
    
    # Header
    console.print(Panel(
        f"[bold]{package.name}[/] v{package.version}\n\n{package.description or '[dim]No description[/]'}",
        title="[cyan]COP Package[/]",
        border_style="cyan"
    ))
    console.print()
    
    # Package tree
    tree = Tree(f"[bold cyan]{package_path}[/]")
    
    # System prompt
    if package.system_prompt:
        prompt_branch = tree.add("[bold]📝 System Prompt[/]")
        prompt_branch.add(f"[dim]Source:[/] {package.system_prompt.source}")
        if package.system_prompt.variables:
            vars_str = ", ".join(f"{{{{[cyan]{v}[/]}}}}" for v in package.system_prompt.variables)
            prompt_branch.add(f"[dim]Variables:[/] {vars_str}")
    
    # Personas
    if package.personas:
        personas_branch = tree.add("[bold]👤 Personas[/]")
        default_persona = package.default_persona
        for name, persona in package.personas.items():
            marker = " [green](default)[/]" if name == default_persona else ""
            personas_branch.add(f"{name}{marker} - [dim]{persona.source}[/]")
    
    # Guardrails
    if package.guardrails:
        guardrails_branch = tree.add("[bold]🛡️ Guardrails[/]")
        for gr in package.guardrails:
            guardrails_branch.add(f"{gr.name} [dim](priority: {gr.priority})[/]")
    
    # Tools
    if package.tools:
        tools_branch = tree.add("[bold]🔧 Tools[/]")
        for tool in package.tools:
            tools_branch.add(f"{tool.name} - [dim]{tool.source}[/]")
    
    # Knowledge
    if package.knowledge:
        knowledge_branch = tree.add("[bold]📚 Knowledge[/]")
        for k in package.knowledge:
            knowledge_branch.add(f"{k.name} [dim]({k.type})[/]")
    
    console.print(tree)
    console.print()
    
    # Statistics table
    table = Table(title="Statistics", show_header=False, box=None)
    table.add_column(style="bold")
    table.add_column()
    
    table.add_row("Personas:", str(len(package.personas)))
    table.add_row("Guardrails:", str(len(package.guardrails)))
    table.add_row("Tools:", str(len(package.tools)))
    table.add_row("Knowledge sources:", str(len(package.knowledge)))
    
    if package.system_prompt:
        lines = len(package.system_prompt.content.split('\n'))
        table.add_row("System prompt lines:", str(lines))
    
    console.print(table)
    console.print()
    
    return True

