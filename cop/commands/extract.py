"""
COP Extract Command - Extract .ftpack artifacts
"""

import json
import tarfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree


def run_extract(
    ftpack_path: Path,
    output_dir: Path = None,
    list_only: bool = False,
    console: Console = None
) -> bool:
    """Extract a .ftpack artifact.
    
    Args:
        ftpack_path: Path to the .ftpack file
        output_dir: Directory to extract to (default: current dir)
        list_only: Just list contents, don't extract
        console: Rich console for output
    
    Returns:
        True if successful, False otherwise
    """
    console = console or Console()
    
    if not ftpack_path.exists():
        console.print(f"[red]Error:[/] File not found: {ftpack_path}")
        return False
    
    if not str(ftpack_path).endswith(".ftpack"):
        console.print(f"[yellow]Warning:[/] File doesn't have .ftpack extension")
    
    try:
        with tarfile.open(ftpack_path, "r:gz") as tar:
            if list_only:
                # List contents
                console.print()
                console.print(Panel(
                    f"[bold]{ftpack_path.name}[/]",
                    title="[cyan]FTPack Contents[/]",
                    border_style="cyan"
                ))
                console.print()
                
                tree = Tree(f"[bold cyan]{ftpack_path.name}[/]")
                
                # Read manifest if available
                manifest = None
                try:
                    manifest_file = tar.extractfile("manifest.json")
                    if manifest_file:
                        manifest = json.load(manifest_file)
                except:
                    pass
                
                for member in tar.getmembers():
                    size_str = f"[dim]({member.size:,} bytes)[/]"
                    if member.isdir():
                        tree.add(f"📁 {member.name}/")
                    else:
                        tree.add(f"📄 {member.name} {size_str}")
                
                console.print(tree)
                console.print()
                
                if manifest:
                    console.print(f"[bold]Package:[/] {manifest.get('name', 'unknown')}")
                    console.print(f"[bold]Version:[/] {manifest.get('package_version', 'unknown')}")
                    console.print(f"[bold]Created:[/] {manifest.get('created_at', 'unknown')}")
                    console.print()
                
                return True
            
            # Extract
            output_dir = output_dir or Path(".")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            console.print()
            console.print(Panel(
                f"[bold]Extracting:[/] {ftpack_path.name}\n[bold]To:[/] {output_dir}",
                title="[cyan]COP Extract[/]",
                border_style="cyan"
            ))
            console.print()
            
            tar.extractall(output_dir)
            
            for member in tar.getmembers():
                console.print(f"  [green]✓[/] {member.name}")
            
            console.print()
            console.print(f"[bold green]✓ Extracted {len(tar.getmembers())} files[/]")
            console.print()
            
            return True
            
    except tarfile.TarError as e:
        console.print(f"[red]Error:[/] Failed to read archive: {e}")
        return False
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/] Archive not found: {e}")
        return False
    except PermissionError as e:
        console.print(f"[red]Error:[/] Permission denied: {e}")
        return False
    except (IOError, OSError) as e:
        console.print(f"[red]Error:[/] File system error: {e}")
        return False

