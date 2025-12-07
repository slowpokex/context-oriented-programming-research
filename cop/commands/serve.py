"""
COP Serve Command - Start local development server
"""

from pathlib import Path
from typing import Optional

import yaml
from openai import APIError, APIConnectionError
from rich.console import Console
from rich.panel import Panel

from cop.pipeline.constants import DEFAULT_LLM_ENDPOINT


def run_serve(
    port: int = 8080,
    lm_studio_url: str = DEFAULT_LLM_ENDPOINT,
    package_path: Optional[Path] = None,
    console: Console = None
):
    """Start local development server with LM Studio integration.
    
    Args:
        port: Server port
        lm_studio_url: LM Studio API URL
        package_path: COP package to serve
        console: Rich console for output
    """
    console = console or Console()
    
    console.print()
    console.print(Panel(
        f"[bold]Starting COP Development Server[/]\n\n"
        f"Port: [cyan]{port}[/]\n"
        f"LM Studio: [cyan]{lm_studio_url}[/]\n"
        f"Package: [cyan]{package_path or 'None (interactive mode)'}[/]",
        title="[cyan]COP Serve[/]",
        border_style="cyan"
    ))
    console.print()
    
    # Check LM Studio connectivity
    console.print("[dim]Checking LM Studio connectivity...[/]")
    
    try:
        from openai import OpenAI
        client = OpenAI(base_url=lm_studio_url, api_key="not-needed")
        models = client.models.list()
        
        if models.data:
            console.print(f"[green]✓[/] Connected to LM Studio")
            console.print(f"  Available models: {', '.join(m.id for m in models.data[:3])}")
        else:
            console.print("[yellow]⚠[/] LM Studio connected but no models loaded")
            console.print("  Load a model in LM Studio to enable inference")
    except ImportError:
        console.print("[red]✗[/] OpenAI package not installed")
        console.print("  Run: pip install openai")
        return
    except (APIError, APIConnectionError) as e:
        console.print(f"[red]✗[/] Could not connect to LM Studio: {e}")
        console.print("  Make sure LM Studio is running with the server enabled")
        return
    except (ConnectionError, TimeoutError, OSError) as e:
        console.print(f"[red]✗[/] Network error connecting to LM Studio: {e}")
        console.print("  Check your network connection and LM Studio server")
        return
    
    console.print()
    
    # Load package if specified
    if package_path:
        try:
            from cop.core.package import COPPackage
            package = COPPackage.load(package_path)
            console.print(f"[green]✓[/] Loaded package: {package.name} v{package.version}")
        except FileNotFoundError as e:
            console.print(f"[red]✗[/] Package not found: {e}")
            return
        except (yaml.YAMLError, ValueError) as e:
            console.print(f"[red]✗[/] Invalid package format: {e}")
            return
        except (IOError, OSError) as e:
            console.print(f"[red]✗[/] Failed to read package: {e}")
            return
    
    console.print()
    console.print("[yellow]Note:[/] Full server implementation coming soon.")
    console.print()
    console.print("For now, you can use the LM Studio API directly at:")
    console.print(f"  [cyan]{lm_studio_url}[/]")
    console.print()
    console.print("Example usage:")
    console.print(f'''
    from openai import OpenAI
    client = OpenAI(base_url="{lm_studio_url}", api_key="x")
    
    response = client.chat.completions.create(
        model="local-model",
        messages=[{{"role": "user", "content": "Hello!"}}]
    )
    print(response.choices[0].message.content)
    ''')

