"""
COP Validate Command - Validate COP packages against schemas
"""

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cop.core.validator import COPValidator, Severity


def run_validate(
    package_path: Path,
    verbose: bool = False,
    strict: bool = False,
    output_format: str = "text",
    console: Console = None
) -> bool:
    """Run validation on a COP package.
    
    Args:
        package_path: Path to the COP package directory
        verbose: Show detailed output
        strict: Treat warnings as errors
        output_format: Output format (text or json)
        console: Rich console for output
    
    Returns:
        True if validation passed, False otherwise
    """
    console = console or Console()
    
    validator = COPValidator()
    result = validator.validate_package(package_path)
    
    if output_format == "json":
        console.print_json(data=result.to_dict())
        return result.is_valid and (not strict or not result.has_warnings)
    
    # Text output
    console.print()
    console.print(Panel(
        f"[bold]Validating:[/] {package_path}",
        title="[cyan]COP Validation[/]",
        border_style="cyan"
    ))
    console.print()
    
    if not result.issues:
        console.print("[bold green]✓ All validations passed![/]")
        console.print(f"  Files validated: {result.files_validated}")
        return True
    
    # Group issues by file
    issues_by_file: dict = {}
    for issue in result.issues:
        if issue.file_path not in issues_by_file:
            issues_by_file[issue.file_path] = []
        issues_by_file[issue.file_path].append(issue)
    
    # Display issues
    for file_path, issues in issues_by_file.items():
        console.print(f"[bold cyan]{file_path}[/]")
        
        for issue in issues:
            severity_style = {
                Severity.ERROR: "red",
                Severity.WARNING: "yellow",
                Severity.INFO: "blue",
            }.get(issue.severity, "white")
            
            console.print(f"  [{severity_style}][{issue.severity.value.upper()}][/] {issue.message}")
            
            if verbose:
                if issue.json_path:
                    console.print(f"    [dim]Path: {issue.json_path}[/]")
                if issue.suggestion:
                    console.print(f"    [green]Suggestion: {issue.suggestion}[/]")
        
        console.print()
    
    # Summary
    table = Table(title="Summary", show_header=False, box=None)
    table.add_column(style="bold")
    table.add_column()
    
    table.add_row("Errors:", f"[red]{result.error_count}[/]")
    table.add_row("Warnings:", f"[yellow]{result.warning_count}[/]")
    table.add_row("Files validated:", str(result.files_validated))
    
    console.print(table)
    
    passed = result.is_valid and (not strict or not result.has_warnings)
    
    if passed:
        console.print("\n[bold green]✓ Validation passed[/]")
    else:
        console.print("\n[bold red]✗ Validation failed[/]")
    
    return passed

