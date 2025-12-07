"""
COP Run Command - Interactive chat with a COP agent

Runs an interactive REPL session using the compiled COP package
with optional RAG-enhanced context retrieval.

Supports multiple providers:
  - Local: LM Studio, Ollama, vLLM, etc.
  - OpenAI: gpt-4, gpt-3.5-turbo, etc.
  - OpenRouter: openai/gpt-4, anthropic/claude-3, etc.
  - Any OpenAI-compatible API

MCP Integration:
  Configure MCP servers in cop.yaml to give your agent access to
  external tools like filesystem, browser, databases, etc.
"""

import asyncio
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from cop.pipeline.constants import DEFAULT_LLM_ENDPOINT, DEFAULT_LLM_MODEL, DEFAULT_API_KEY


# Known provider endpoints
PROVIDER_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com/v1",  # Note: uses different API format
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "local": DEFAULT_LLM_ENDPOINT,
    "lmstudio": "http://localhost:1234/v1",
    "ollama": "http://localhost:11434/v1",
}

# Environment variable names for API keys per provider
PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
}


def run_interactive(
    package_path: Path,
    persona: Optional[str] = None,
    endpoint: str = None,
    provider: str = None,
    model: str = None,
    api_key: str = None,
    use_rag: bool = True,
    use_tools: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_history: int = 20,
    verbose: bool = False,
    console: Console = None
) -> bool:
    """Run an interactive chat session with a COP agent.
    
    Args:
        package_path: Path to COP package directory or .ftpack file
        persona: Persona to use (default from package)
        endpoint: LLM API endpoint (auto-detected from provider if not set)
        provider: Provider name (openai, openrouter, local, etc.)
        model: Model name (e.g., gpt-4, openai/gpt-4o, local-model)
        api_key: API key (uses env var if not provided)
        use_rag: Enable RAG context retrieval
        use_tools: Enable tool calling (requires model support)
        temperature: Response temperature
        max_tokens: Max response tokens
        max_history: Maximum conversation turns to keep (prevents context overflow)
        verbose: Show detailed debug output
        console: Rich console for output
    
    Returns:
        True if session ended normally, False on error
    """
    console = console or Console()
    
    def vlog(message: str) -> None:
        """Print verbose log message."""
        if verbose:
            console.print(f"[dim]DEBUG: {message}[/]")
    
    # Import OpenAI client
    try:
        from openai import OpenAI, APIError, APIConnectionError
    except ImportError:
        console.print("[red]Error:[/] OpenAI package not installed")
        console.print("  Run: pip install openai")
        return False
    
    # Resolve provider, endpoint, model, and API key
    resolved = _resolve_provider_config(
        provider=provider,
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        console=console
    )
    if resolved is None:
        return False
    
    final_endpoint, final_model, final_api_key, provider_name = resolved
    vlog(f"Provider resolved: {provider_name} @ {final_endpoint}")
    vlog(f"Model: {final_model}")
    
    # Load package
    vlog(f"Loading package from: {package_path}")
    package_data = _load_package(package_path, console)
    if package_data is None:
        return False
    
    system_prompt = package_data.get("system_prompt", "")
    package_name = package_data.get("name", "COP Agent")
    package_version = package_data.get("version", "0.0.0")
    personas = package_data.get("personas", {})
    tools = package_data.get("tools", [])
    guardrails = package_data.get("guardrails", [])
    vector_index = package_data.get("vector_index")
    
    vlog(f"Package: {package_name} v{package_version}")
    vlog(f"System prompt length: {len(system_prompt)} chars")
    vlog(f"Personas: {list(personas.keys())}")
    vlog(f"Guardrails: {[g.get('name') for g in guardrails]}")
    vlog(f"Tools: {[t.get('name') for t in tools]}")
    vlog(f"Vector index: {'loaded' if vector_index else 'not available'}")
    
    # Select persona
    active_persona = None
    if persona and persona in personas:
        active_persona = personas[persona]
        console.print(f"[dim]Using persona: {persona}[/]")
    elif personas:
        default_persona = package_data.get("default_persona")
        if default_persona and default_persona in personas:
            active_persona = personas[default_persona]
            console.print(f"[dim]Using default persona: {default_persona}[/]")
    
    # Enhance system prompt with persona
    if active_persona:
        persona_instruction = _format_persona(active_persona)
        system_prompt = f"{system_prompt}\n\n{persona_instruction}"
    
    # Enhance system prompt with guardrails
    if guardrails:
        guardrail_instruction = _format_guardrails(guardrails)
        system_prompt = f"{system_prompt}\n\n{guardrail_instruction}"
        console.print(f"[dim]Loaded {len(guardrails)} guardrails[/]")
    
    # Connect to LLM
    console.print(f"[dim]Connecting to {provider_name}...[/]")
    try:
        client = OpenAI(base_url=final_endpoint, api_key=final_api_key)
        
        # For local providers, auto-detect model if not specified
        if provider_name in ("local", "lmstudio", "ollama"):
            try:
                models_list = client.models.list()
                if models_list.data:
                    # Auto-detect model if not explicitly set
                    if not final_model:
                        final_model = models_list.data[0].id
                        console.print(f"[dim]Auto-detected model: {final_model}[/]")
                else:
                    console.print("[yellow]Warning:[/] No models loaded in LM Studio")
                    console.print("  Load a model in LM Studio first")
                    return False
            except (APIError, APIConnectionError) as e:
                console.print(f"[yellow]Warning:[/] Could not list models (API error): {e}")
                if not final_model:
                    console.print("[red]Error:[/] No model specified and auto-detect failed")
                    return False
            except (ConnectionError, TimeoutError, OSError) as e:
                console.print(f"[yellow]Warning:[/] Could not connect to list models: {e}")
                if not final_model:
                    console.print("[red]Error:[/] No model specified and auto-detect failed")
                    return False
                
    except (APIError, APIConnectionError, ConnectionError, OSError) as e:
        console.print(f"[red]Error:[/] Could not connect to {provider_name}: {e}")
        return False
    
    # Build trigger phrases for guardrail validation
    trigger_phrases = _extract_trigger_phrases(guardrails)
    violation_responses = _extract_violation_responses(guardrails)
    
    # Setup MCP tools
    mcp_manager = None
    mcp_tools_openai = []
    mcp_config = package_data.get("mcp_config", {})
    
    if mcp_config.get("servers"):
        vlog(f"MCP: Setting up {len(mcp_config['servers'])} server(s)...")
        try:
            from cop.mcp_client import setup_mcp_tools
            mcp_manager = asyncio.run(setup_mcp_tools(
                package_data.get("manifest", {}),
                console,
                verbose=verbose
            ))
            if mcp_manager:
                mcp_tools_openai = mcp_manager.get_openai_tools()
                vlog(f"MCP: {len(mcp_tools_openai)} tools available")
                
                # Add MCP system instructions to prompt if configured
                mcp_instructions = mcp_config.get("system_instructions", "")
                if mcp_instructions:
                    system_prompt = f"{system_prompt}\n\n[MCP TOOLS]\n{mcp_instructions}"
                    vlog("MCP: Added system instructions to prompt")
        except ImportError as e:
            vlog(f"MCP: Could not import mcp_client: {e}")
        except Exception as e:
            console.print(f"[yellow]MCP setup error:[/] {e}")
            if verbose:
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/]")
    
    # Convert COP tools to OpenAI function format (if tools enabled)
    cop_tools_openai = []
    all_tools_openai = []
    
    if use_tools:
        for t in tools:
            definition = t.get("definition", {})
            cop_tools_openai.append({
                "type": "function",
                "function": {
                    "name": t.get("name", "unknown"),
                    "description": definition.get("description", ""),
                    "parameters": definition.get("parameters", {"type": "object", "properties": {}}),
                }
            })
        
        # Combine COP tools with MCP tools
        all_tools_openai = cop_tools_openai + mcp_tools_openai
        vlog(f"Tools enabled: {len(cop_tools_openai)} COP + {len(mcp_tools_openai)} MCP")
    else:
        vlog("Tools disabled (--no-tools)")
    
    total_tools = len(tools) + len(mcp_tools_openai)
    
    # Show header
    console.print()
    mcp_status = f", MCP: {len(mcp_tools_openai)} tools" if mcp_tools_openai else ""
    console.print(Panel(
        f"[bold]{package_name}[/] v{package_version}\n\n"
        f"[dim]Provider:[/] {provider_name}\n"
        f"[dim]Model:[/] {final_model}\n"
        f"[dim]RAG:[/] {'enabled' if use_rag and vector_index else 'disabled'}\n"
        f"[dim]Guardrails:[/] {len(guardrails)} active\n"
        f"[dim]Tools:[/] {len(tools)} COP{mcp_status}\n\n"
        f"[yellow]Commands:[/] /quit, /clear, /persona <name>, /rag on|off, /tools, /help",
        title="[cyan]COP Interactive Session[/]",
        border_style="cyan"
    ))
    console.print()
    
    # Chat history
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    
    rag_enabled = use_rag and vector_index is not None
    
    # REPL loop
    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            break
        
        user_input = user_input.strip()
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            # Handle /tools specially since it needs mcp_tools
            if user_input.lower().strip() in ("/tools", "/tool"):
                console.print("[bold]Available Tools:[/]")
                if tools:
                    console.print("\n[cyan]COP Tools:[/]")
                    for t in tools:
                        console.print(f"  • {t.get('name', 'unknown')}")
                if mcp_tools_openai:
                    console.print("\n[cyan]MCP Tools:[/]")
                    for t in mcp_tools_openai:
                        name = t["function"]["name"]
                        desc = t["function"]["description"][:60] + "..." if len(t["function"]["description"]) > 60 else t["function"]["description"]
                        console.print(f"  • {name}")
                        console.print(f"    [dim]{desc}[/]")
                if not tools and not mcp_tools_openai:
                    console.print("[dim]No tools available[/]")
                continue
            
            cmd_result = _handle_command(
                user_input, 
                messages, 
                personas, 
                rag_enabled,
                console
            )
            if cmd_result == "quit":
                break
            elif cmd_result == "clear":
                messages = [{"role": "system", "content": system_prompt}]
                console.print("[dim]Chat history cleared.[/]")
            elif cmd_result == "rag_on":
                rag_enabled = vector_index is not None
                console.print(f"[dim]RAG: {'enabled' if rag_enabled else 'no index available'}[/]")
            elif cmd_result == "rag_off":
                rag_enabled = False
                console.print("[dim]RAG: disabled[/]")
            elif cmd_result and cmd_result.startswith("persona:"):
                new_persona = cmd_result.split(":", 1)[1]
                if new_persona in personas:
                    active_persona = personas[new_persona]
                    persona_instruction = _format_persona(active_persona)
                    new_system = f"{package_data.get('system_prompt', '')}\n\n{persona_instruction}"
                    messages[0] = {"role": "system", "content": new_system}
                    console.print(f"[dim]Switched to persona: {new_persona}[/]")
                else:
                    console.print(f"[yellow]Unknown persona.[/] Available: {', '.join(personas.keys())}")
            continue
        
        # Check input against guardrail scope constraints (keyword-based)
        if guardrails:
            vlog(f"Checking input against {len(guardrails)} guardrails (keywords)...")
            rejection = _check_input_guardrails(user_input, guardrails)
            if rejection:
                vlog(f"Input REJECTED by guardrail: {rejection[:50]}...")
                console.print()
                console.print("[bold yellow]Assistant[/]")
                console.print()
                console.print(rejection)
                console.print()
                continue
            vlog("Input passed keyword guardrail checks")
        
        # RAG context retrieval + semantic off-topic detection
        rag_context = ""
        rag_max_score = 0.0
        if rag_enabled and vector_index:
            vlog("RAG: Retrieving context...")
            embedding_config = package_data.get("embedding_config")
            vlog(f"RAG: Embedding config: {embedding_config}")
            rag_context, rag_max_score = _retrieve_context_with_score(
                user_input, vector_index, console, 
                embedding_config=embedding_config,
                verbose=verbose
            )
            if rag_context:
                vlog(f"RAG: Retrieved {len(rag_context)} chars of context (max_score={rag_max_score:.4f})")
            else:
                vlog(f"RAG: No context retrieved (max_score={rag_max_score:.4f})")
        
        # Semantic off-topic detection using RAG scores
        if guardrails and rag_enabled and vector_index:
            scope_config = _extract_scope_constraints(guardrails)
            min_relevance = scope_config.get("min_relevance_score", 0.55)
            
            # If RAG score is very low, the query is likely off-topic
            is_substantial = len(user_input.split()) > 2
            if is_substantial and rag_max_score < min_relevance:
                vlog(f"Semantic check: score {rag_max_score:.4f} < threshold {min_relevance}")
                off_topic_response = scope_config.get("off_topic_response",
                    "I'm a specialized assistant. Please ask about my area of expertise.")
                console.print()
                console.print("[bold yellow]Assistant[/]")
                console.print()
                console.print(off_topic_response)
                console.print()
                continue
            vlog(f"Semantic check PASSED: score {rag_max_score:.4f} >= threshold {min_relevance}")
        
        # Build message with RAG context
        if rag_context:
            enhanced_input = f"{user_input}\n\n[Relevant Context]\n{rag_context}"
            vlog(f"Enhanced input with RAG context ({len(enhanced_input)} chars)")
        else:
            enhanced_input = user_input
        
        messages.append({"role": "user", "content": enhanced_input})
        vlog(f"Message history: {len(messages)} messages")
        
        # Generate response (with tool calling support)
        assistant_message = ""
        try:
            console.print()
            console.print("[bold green]Assistant[/]")
            console.print()
            
            vlog(f"API call: model={final_model}, temp={temperature}, max_tokens={max_tokens}")
            if all_tools_openai:
                vlog(f"Tools available: {len(all_tools_openai)}")
            
            # Build completion request
            completion_kwargs = {
                "model": final_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Add tools if available (COP + MCP tools)
            if all_tools_openai:
                completion_kwargs["tools"] = all_tools_openai
                completion_kwargs["tool_choice"] = "auto"
            
            # Tool calling loop - keep going until no more tool calls
            while True:
                # Non-streaming for tool calls, streaming for final response
                if all_tools_openai:
                    # Use non-streaming for potential tool calls
                    response = client.chat.completions.create(**completion_kwargs)
                    choice = response.choices[0]
                    
                    # Check if LLM wants to call tools
                    if choice.message.tool_calls:
                        vlog(f"Tool calls requested: {len(choice.message.tool_calls)}")
                        
                        # Add assistant message with tool calls
                        messages.append(choice.message.model_dump())
                        
                        # Execute each tool call
                        for tool_call in choice.message.tool_calls:
                            tool_name = tool_call.function.name
                            try:
                                tool_args = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError:
                                tool_args = {}
                            
                            vlog(f"Calling tool: {tool_name}")
                            console.print(f"[dim]🔧 Using tool: {tool_name}[/]")
                            
                            # Execute tool based on type
                            if mcp_manager and "__" in tool_name:
                                # MCP tool (format: server__tool)
                                try:
                                    result = asyncio.run(mcp_manager.call_tool(tool_name, tool_args))
                                    tool_result = json.dumps(result) if result else "Tool returned no result"
                                except Exception as e:
                                    tool_result = f"Tool error: {e}"
                                    vlog(f"Tool error: {e}")
                            else:
                                # COP tool - execute locally
                                tool_result = _execute_cop_tool(tool_name, tool_args, package_path, console, verbose)
                            
                            # Add tool result to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result,
                            })
                            vlog(f"Tool result: {tool_result[:200]}...")
                        
                        # Continue the loop to let LLM process tool results
                        continue
                    
                    # No tool calls - output the response
                    assistant_message = choice.message.content or ""
                    console.print(assistant_message)
                    break
                    
                else:
                    # No tools - use streaming
                    completion_kwargs["stream"] = True
                    stream = client.chat.completions.create(**completion_kwargs)
                    
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            assistant_message += token
                            console.print(token, end="")
                    break
            
            console.print("\n")  # End the output
            
            # Check guardrails on response
            if trigger_phrases:
                violations = _check_guardrail_violations(assistant_message, trigger_phrases)
                if violations:
                    console.print("[yellow]⚠ Guardrail warning:[/]", style="bold")
                    for violation in violations[:3]:  # Show max 3
                        response = violation_responses.get(violation["type"], 
                            f"Response may contain: {violation['phrase']}")
                        console.print(f"  [dim]• {response}[/]")
                    console.print()
            
            # Add to history
            messages.append({"role": "assistant", "content": assistant_message})
            
            # Truncate history if too long (keep system + last N messages)
            messages = _truncate_history(messages, max_history, console)
            
        except (APIError, APIConnectionError) as e:
            console.print(f"\n[red]API Error:[/] {e}")
            messages.pop()  # Remove failed user message
        except KeyboardInterrupt:
            console.print("\n[dim]Response cancelled.[/]")
            if assistant_message:
                # Keep partial response in history
                messages.append({"role": "assistant", "content": assistant_message + "..."})
            else:
                messages.pop()  # Remove user message if no response started
    
    # Cleanup MCP connections
    if mcp_manager:
        vlog("Disconnecting MCP servers...")
        mcp_manager.disconnect_all()
    
    return True


def _truncate_history(
    messages: List[Dict[str, str]], 
    max_turns: int,
    console: Console
) -> List[Dict[str, str]]:
    """Truncate message history to prevent context overflow.
    
    Keeps:
    - System message (always first)
    - Last max_turns * 2 messages (user + assistant pairs)
    
    Args:
        messages: Full message history
        max_turns: Max conversation turns to keep
        console: Console for output
    
    Returns:
        Truncated message list
    """
    if len(messages) <= 1:
        return messages
    
    # System message + max_turns pairs (user + assistant)
    max_messages = 1 + (max_turns * 2)
    
    if len(messages) <= max_messages:
        return messages
    
    # Keep system message + last N messages
    system_msg = messages[0]
    recent_msgs = messages[-(max_turns * 2):]
    
    truncated = [system_msg] + recent_msgs
    
    dropped = len(messages) - len(truncated)
    console.print(f"[dim](Truncated {dropped} old messages to stay within context limit)[/]")
    
    return truncated


def _resolve_provider_config(
    provider: Optional[str],
    endpoint: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    console: Console
) -> Optional[tuple]:
    """Resolve provider configuration from arguments and environment.
    
    Resolution priority:
    1. Explicit arguments
    2. Environment variables  
    3. Defaults (local LM Studio)
    
    Note: Model names with slashes (e.g., openai/gpt-oss-20b) are passed through
    as-is for local providers. Use --provider openrouter explicitly for OpenRouter.
    
    Returns:
        Tuple of (endpoint, model, api_key, provider_name) or None on error
    """
    # Start with defaults
    final_provider = provider
    final_endpoint = endpoint
    final_model = model  # None means auto-detect for local providers
    final_api_key = api_key
    
    # Default to local if no provider specified
    # Model names like "openai/gpt-oss-20b" are valid LM Studio model names
    if not final_provider:
        final_provider = "local"
    
    # Resolve endpoint from provider
    if not final_endpoint:
        if final_provider in PROVIDER_ENDPOINTS:
            final_endpoint = PROVIDER_ENDPOINTS[final_provider]
        else:
            # Treat unknown provider as custom endpoint
            final_endpoint = final_provider
            final_provider = "custom"
    
    # Resolve API key
    if not final_api_key:
        # Check provider-specific env var
        if final_provider in PROVIDER_API_KEY_ENV:
            env_var = PROVIDER_API_KEY_ENV[final_provider]
            final_api_key = os.environ.get(env_var)
            if not final_api_key:
                console.print(f"[red]Error:[/] API key required for {final_provider}")
                console.print(f"  Set {env_var} environment variable or use --api-key")
                return None
        else:
            # Local providers don't need API key
            final_api_key = DEFAULT_API_KEY
    
    # Cloud providers require explicit model
    cloud_providers = {"openai", "openrouter", "groq", "together", "anthropic"}
    if final_provider in cloud_providers and not final_model:
        console.print(f"[red]Error:[/] Model required for {final_provider}")
        console.print(f"  Use --model to specify (e.g., --model gpt-4o)")
        return None
    
    return (final_endpoint, final_model, final_api_key, final_provider)


def _load_package(package_path: Path, console: Console) -> Optional[Dict[str, Any]]:
    """Load package from directory or .ftpack file."""
    package_path = Path(package_path)
    
    # Handle .ftpack files
    if package_path.suffix == ".ftpack" or str(package_path).endswith(".ftpack"):
        return _load_ftpack(package_path, console)
    
    # Handle package directory
    if package_path.is_dir():
        return _load_package_dir(package_path, console)
    
    # Try as .ftpack anyway
    if package_path.exists():
        return _load_ftpack(package_path, console)
    
    console.print(f"[red]Error:[/] Package not found: {package_path}")
    return None


def _load_package_dir(package_dir: Path, console: Console) -> Optional[Dict[str, Any]]:
    """Load package from directory."""
    try:
        from cop.core.package import COPPackage
        
        package = COPPackage.load(package_dir)
        
        # Compile system prompt with default variables
        system_prompt = ""
        if package.system_prompt:
            # Get default variable values from manifest
            context = package.manifest.get("context", {})
            system_config = context.get("system", {})
            var_defs = system_config.get("variables", {})
            defaults = {k: v.get("default", "") for k, v in var_defs.items() if isinstance(v, dict)}
            system_prompt = package.compile_prompt(defaults)
        
        # Load personas
        personas = {}
        if package.personas:
            for name, p in package.personas.items():
                personas[name] = p.data
        default_persona = package.default_persona
        
        # Load guardrails
        guardrails = [{"name": g.name, "priority": g.priority, "data": g.data} 
                      for g in package.guardrails]
        
        # Load tools
        tools = [{"name": t.name, "definition": t.data} for t in package.tools]
        
        # Try to load vector index from dist/
        vector_index = _try_load_index(package_dir / "dist" / "embeddings", console)
        
        # Get embedding config from manifest for RAG queries
        build_config = package.manifest.get("build", {})
        linking_config = build_config.get("linking", {})
        local_llm_config = build_config.get("local_llm", {})
        embedding_config = {
            "endpoint": linking_config.get("endpoint", local_llm_config.get("endpoint", "http://localhost:1234/v1")),
            "model": linking_config.get("embedding_model", "text-embedding-nomic-embed-text-v1.5"),
            "api_key": linking_config.get("api_key", "not-needed"),
        }
        
        # Get MCP configuration
        mcp_config = package.manifest.get("mcp", {})
        
        return {
            "name": package.name,
            "version": package.version,
            "system_prompt": system_prompt,
            "personas": personas,
            "default_persona": default_persona,
            "guardrails": guardrails,
            "tools": tools,
            "vector_index": vector_index,
            "embedding_config": embedding_config,
            "mcp_config": mcp_config,
            "manifest": package.manifest,
        }
        
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/] Package file not found: {e}")
        return None
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        console.print(f"[red]Error:[/] Invalid package format: {e}")
        return None
    except (ValueError, KeyError, TypeError) as e:
        console.print(f"[red]Error loading package:[/] {e}")
        return None
    except (IOError, OSError) as e:
        console.print(f"[red]Error reading package:[/] {e}")
        return None


def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    """Safely extract tarfile, preventing path traversal attacks.
    
    Validates that all extracted files stay within the destination directory.
    Rejects absolute paths, paths with .. components, and symbolic links
    pointing outside the destination.
    
    Args:
        tar: Open tarfile object
        dest: Destination directory (must exist)
    
    Raises:
        ValueError: If a member would extract outside dest
    """
    dest = dest.resolve()
    
    for member in tar.getmembers():
        # Get the target path
        member_path = (dest / member.name).resolve()
        
        # Verify it's within the destination
        try:
            member_path.relative_to(dest)
        except ValueError:
            raise ValueError(
                f"Attempted path traversal in archive: {member.name}"
            )
        
        # Reject absolute paths in the archive
        if member.name.startswith('/') or member.name.startswith('\\'):
            raise ValueError(
                f"Absolute path in archive: {member.name}"
            )
        
        # Reject suspicious components
        if '..' in member.name.split('/') or '..' in member.name.split('\\'):
            raise ValueError(
                f"Path traversal component in archive: {member.name}"
            )
    
    # All members validated, safe to extract
    tar.extractall(dest)


def _load_ftpack(ftpack_path: Path, console: Console) -> Optional[Dict[str, Any]]:
    """Load package from .ftpack archive."""
    try:
        with tarfile.open(ftpack_path, "r:gz") as tar:
            # Extract to temp directory (with path traversal protection)
            with tempfile.TemporaryDirectory() as tmpdir:
                _safe_extract_tar(tar, Path(tmpdir))
                tmpdir_path = Path(tmpdir)
                
                # Load manifest
                manifest_path = tmpdir_path / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                else:
                    manifest = {}
                
                # Load context bundle
                bundle_path = tmpdir_path / "context.bundle.json"
                if bundle_path.exists():
                    with open(bundle_path) as f:
                        bundle = json.load(f)
                else:
                    bundle = {}
                
                # Extract system prompt
                system_prompt = bundle.get("compiled_prompt", "")
                
                # Extract personas
                personas = bundle.get("personas", {})
                default_persona = bundle.get("default_persona")
                
                # Extract guardrails
                guardrails = bundle.get("guardrails", [])
                
                # Extract tools
                tools = bundle.get("tools", [])
                
                # Vector index not available from ftpack (would need separate loading)
                vector_index = None
                
                return {
                    "name": manifest.get("name", "COP Agent"),
                    "version": manifest.get("version", "0.0.0"),
                    "system_prompt": system_prompt,
                    "personas": personas,
                    "default_persona": default_persona,
                    "guardrails": guardrails,
                    "tools": tools,
                    "vector_index": vector_index
                }
                
    except tarfile.TarError as e:
        console.print(f"[red]Error reading .ftpack:[/] {e}")
        return None
    except (json.JSONDecodeError, IOError) as e:
        console.print(f"[red]Error parsing .ftpack contents:[/] {e}")
        return None


def _try_load_index(embeddings_dir: Path, console: Console) -> Optional[Any]:
    """Try to load vector index from embeddings directory."""
    if not embeddings_dir.exists():
        return None
    
    try:
        from cop.pipeline.indexer import VectorIndex
        
        index = VectorIndex.load(embeddings_dir)
        if index and len(index.chunk_ids) > 0:
            console.print(f"[dim]Loaded RAG index with {len(index.chunk_ids)} chunks[/]")
            return index
    except ImportError:
        console.print("[dim]RAG index not loaded: indexer module unavailable[/]")
    except FileNotFoundError:
        pass  # No index file, that's fine
    except (json.JSONDecodeError, ValueError) as e:
        console.print(f"[dim]RAG index not loaded: invalid format ({e})[/]")
    except (IOError, OSError) as e:
        console.print(f"[dim]RAG index not loaded: read error ({e})[/]")
    
    return None


def _format_persona(persona: Dict[str, Any]) -> str:
    """Format persona data as system prompt addition."""
    parts = []
    
    if persona.get("tone"):
        parts.append(f"Communication tone: {persona['tone']}")
    
    if persona.get("style"):
        parts.append(f"Response style: {persona['style']}")
    
    if persona.get("characteristics"):
        chars = ", ".join(persona["characteristics"])
        parts.append(f"Key characteristics: {chars}")
    
    vocab = persona.get("vocabulary", {})
    if vocab.get("preferred"):
        parts.append(f"Preferred phrases: {', '.join(vocab['preferred'][:3])}")
    if vocab.get("avoided"):
        parts.append(f"Avoid: {', '.join(vocab['avoided'][:3])}")
    
    if parts:
        return "[Persona Instructions]\n" + "\n".join(parts)
    return ""


def _retrieve_context(
    query: str, 
    vector_index: Any, 
    console: Console, 
    top_k: int = 3,
    embedding_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> str:
    """Retrieve relevant context using RAG.
    
    Args:
        query: User query text
        vector_index: VectorIndex object with embeddings
        console: Rich console
        top_k: Number of results to retrieve
        embedding_config: Config for embedding generation (endpoint, model)
        verbose: Show detailed debug output
    
    Returns:
        Context string (empty if none found)
    """
    context, _ = _retrieve_context_with_score(
        query, vector_index, console, top_k, embedding_config, verbose
    )
    return context


def _retrieve_context_with_score(
    query: str, 
    vector_index: Any, 
    console: Console, 
    top_k: int = 3,
    embedding_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> tuple[str, float]:
    """Retrieve relevant context using RAG, returning max score.
    
    Args:
        query: User query text
        vector_index: VectorIndex object with embeddings
        console: Rich console
        top_k: Number of results to retrieve
        embedding_config: Config for embedding generation (endpoint, model)
        verbose: Show detailed debug output
    
    Returns:
        Tuple of (context_string, max_similarity_score)
    """
    def vlog(msg: str) -> None:
        if verbose:
            console.print(f"[dim]DEBUG RAG: {msg}[/]")
    
    try:
        vlog(f"Query: '{query[:50]}...' (top_k={top_k})")
        
        # Generate embedding for the query
        query_embedding = _embed_query(query, embedding_config, console, verbose=verbose)
        if query_embedding is None:
            vlog("Failed to generate query embedding")
            return "", 0.0
        
        vlog(f"Query embedding: {len(query_embedding)} dimensions")
        
        results = vector_index.search(query_embedding, top_k=top_k)
        vlog(f"Search returned {len(results)} results")
        
        if not results:
            return "", 0.0
        
        # Get max score for semantic relevance check
        max_score = max(r.score for r in results) if results else 0.0
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.chunk.metadata.get("source", "unknown")
            content = result.chunk.content[:500]  # Truncate
            vlog(f"  [{i}] score={result.score:.4f} source={source}")
            context_parts.append(f"[{i}] From {source}:\n{content}")
        
        return "\n\n".join(context_parts), max_score
        
    except (ValueError, KeyError, TypeError) as e:
        console.print(f"[dim]RAG retrieval error: {e}[/]")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/]")
        return "", 0.0
    except (AttributeError, IndexError) as e:
        console.print(f"[dim]RAG result parsing error: {e}[/]")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/]")
        return "", 0.0


# Global cache for query embeddings during session
_query_embedding_cache: Dict[str, List[float]] = {}


def _embed_query(
    query: str,
    embedding_config: Optional[Dict[str, Any]],
    console: Console,
    verbose: bool = False
) -> Optional[List[float]]:
    """Generate embedding for a query string.
    
    Uses cached embedding if available to avoid repeated API calls.
    """
    global _query_embedding_cache
    
    def vlog(msg: str) -> None:
        if verbose:
            console.print(f"[dim]DEBUG EMBED: {msg}[/]")
    
    # Check cache first
    cache_key = query.strip().lower()
    if cache_key in _query_embedding_cache:
        vlog(f"Cache HIT for query")
        return _query_embedding_cache[cache_key]
    
    vlog(f"Cache MISS - generating embedding")
    
    # Get embedding config (with defaults)
    config = embedding_config or {}
    endpoint = config.get("endpoint", "http://localhost:1234/v1")
    model = config.get("model", "text-embedding-nomic-embed-text-v1.5")
    api_key = config.get("api_key", "not-needed")
    
    vlog(f"Endpoint: {endpoint}")
    vlog(f"Model: {model}")
    
    try:
        from openai import OpenAI
        import time
        
        start = time.time()
        client = OpenAI(base_url=endpoint, api_key=api_key)
        response = client.embeddings.create(
            model=model,
            input=query
        )
        elapsed = (time.time() - start) * 1000
        
        embedding = response.data[0].embedding
        vlog(f"Embedding generated: {len(embedding)}d in {elapsed:.0f}ms")
        
        # Cache for future use
        _query_embedding_cache[cache_key] = embedding
        
        return embedding
        
    except ImportError:
        console.print("[dim]RAG: OpenAI package not available[/]")
        return None
    except Exception as e:
        console.print(f"[dim]RAG embedding error: {e}[/]")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/]")
        return None


def _handle_command(
    cmd: str, 
    messages: List[Dict], 
    personas: Dict,
    rag_enabled: bool,
    console: Console
) -> Optional[str]:
    """Handle slash commands."""
    cmd = cmd.lower().strip()
    
    if cmd in ("/quit", "/exit", "/q"):
        console.print("[dim]Goodbye![/]")
        return "quit"
    
    if cmd in ("/clear", "/reset"):
        return "clear"
    
    if cmd == "/rag on":
        return "rag_on"
    
    if cmd == "/rag off":
        return "rag_off"
    
    if cmd.startswith("/persona "):
        persona_name = cmd.split(" ", 1)[1].strip()
        return f"persona:{persona_name}"
    
    if cmd in ("/personas", "/persona"):
        if personas:
            console.print(f"[dim]Available personas:[/] {', '.join(personas.keys())}")
        else:
            console.print("[dim]No personas available[/]")
        return None
    
    if cmd in ("/help", "/?"):
        console.print(Panel(
            "[bold]Commands:[/]\n\n"
            "/quit, /exit, /q  - End session\n"
            "/clear, /reset    - Clear chat history\n"
            "/persona <name>   - Switch persona\n"
            "/personas         - List available personas\n"
            "/rag on|off       - Toggle RAG context\n"
            "/tools            - List available tools (COP + MCP)\n"
            "/help             - Show this help",
            title="[cyan]Help[/]",
            border_style="dim"
        ))
        return None
    
    if cmd.startswith("/"):
        console.print(f"[yellow]Unknown command:[/] {cmd}")
        console.print("[dim]Type /help for available commands[/]")
        return None
    
    return None


def _format_guardrails(guardrails: List[Dict[str, Any]]) -> str:
    """Format guardrails as system prompt instructions."""
    if not guardrails:
        return ""
    
    parts = ["[GUARDRAILS - You MUST follow these rules]"]
    
    for gr in sorted(guardrails, key=lambda g: g.get("priority", 50), reverse=True):
        data = gr.get("data", {})
        name = gr.get("name", "unnamed")
        
        # Hard constraints (must follow)
        hard = data.get("hard_constraints", [])
        if hard:
            parts.append(f"\n## {name.upper()} (Priority: {gr.get('priority', 50)})")
            parts.append("HARD CONSTRAINTS (never violate):")
            for constraint in hard:
                if isinstance(constraint, dict):
                    desc = constraint.get("description", constraint.get("name", ""))
                    rules = constraint.get("rules", [])
                    if desc:
                        parts.append(f"  - {desc}")
                    for rule in rules[:3]:  # Limit rules per constraint
                        parts.append(f"    • {rule}")
                else:
                    parts.append(f"  - {constraint}")
        
        # Soft constraints (should follow)
        soft = data.get("soft_constraints", [])
        if soft:
            parts.append("\nSOFT CONSTRAINTS (prefer to follow):")
            for constraint in soft[:5]:  # Limit soft constraints
                if isinstance(constraint, dict):
                    desc = constraint.get("description", constraint.get("name", ""))
                    if desc:
                        parts.append(f"  - {desc}")
                else:
                    parts.append(f"  - {constraint}")
    
    return "\n".join(parts)


def _extract_trigger_phrases(guardrails: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Extract trigger phrases from guardrails for validation."""
    triggers = {}
    
    for gr in guardrails:
        data = gr.get("data", {})
        
        for constraint in data.get("hard_constraints", []):
            if isinstance(constraint, dict):
                name = constraint.get("name", "unknown")
                phrases = constraint.get("trigger_phrases", [])
                if phrases:
                    triggers[name] = phrases
    
    return triggers


def _extract_violation_responses(guardrails: List[Dict[str, Any]]) -> Dict[str, str]:
    """Extract violation response messages from guardrails."""
    responses = {}
    
    for gr in guardrails:
        data = gr.get("data", {})
        vr = data.get("violation_responses", {})
        responses.update(vr)
    
    return responses


def _check_guardrail_violations(
    text: str, 
    trigger_phrases: Dict[str, List[str]]
) -> List[Dict[str, str]]:
    """Check text for guardrail trigger phrases."""
    violations = []
    text_lower = text.lower()
    
    for constraint_name, phrases in trigger_phrases.items():
        for phrase in phrases:
            if phrase.lower() in text_lower:
                violations.append({
                    "type": constraint_name,
                    "phrase": phrase
                })
                break  # One violation per constraint is enough
    
    return violations


def _check_input_guardrails(
    user_input: str,
    guardrails: List[Dict[str, Any]],
) -> Optional[str]:
    """Check user input against guardrail scope constraints.
    
    Returns a rejection message if input violates scope, None if allowed.
    """
    user_lower = user_input.lower()
    
    for gr in guardrails:
        data = gr.get("data", {})
        
        # Check scope constraints (what the agent should NOT do)
        scope = data.get("scope", {})
        
        # Check blocked topics
        blocked_topics = scope.get("blocked_topics", [])
        for topic in blocked_topics:
            if isinstance(topic, dict):
                keywords = topic.get("keywords", [])
                response = topic.get("response", "I can't help with that topic.")
                for keyword in keywords:
                    if keyword.lower() in user_lower:
                        return response
            elif isinstance(topic, str) and topic.lower() in user_lower:
                return f"I'm not able to help with {topic}. Please ask about my area of expertise."
        
        # Check required context (user must be asking about relevant topics)
        required_context = scope.get("required_context", [])
        if required_context:
            # Check if ANY required context keyword is present
            has_context = any(
                kw.lower() in user_lower 
                for kw in required_context
            )
            # Only enforce if the message is substantial (not a greeting)
            is_substantial = len(user_input.split()) > 3
            if is_substantial and not has_context:
                # Check for obvious off-topic indicators
                off_topic_indicators = [
                    "play", "game", "joke", "story", "poem", "sing", 
                    "weather", "recipe", "movie", "music", "sports",
                    "trivia", "riddle", "puzzle"
                ]
                if any(ind in user_lower for ind in off_topic_indicators):
                    fallback = scope.get("off_topic_response", 
                        "I'm a specialized assistant. Please ask about my area of expertise.")
                    return fallback
    
    return None  # Input allowed


def _execute_cop_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    package_path: Path,
    console: Console,
    verbose: bool = False
) -> str:
    """Execute a COP tool locally.
    
    COP tools like read_file, search_codebase are executed directly.
    """
    def vlog(msg: str) -> None:
        if verbose:
            console.print(f"[dim]TOOL [{tool_name}]: {msg}[/]")
    
    try:
        if tool_name == "read_file":
            # Read a file from the project
            file_path = arguments.get("path") or arguments.get("file_path") or arguments.get("filename")
            if not file_path:
                return "Error: No file path provided"
            
            target = package_path / file_path
            if not target.exists():
                # Try relative to current dir
                target = Path(file_path)
            
            if not target.exists():
                return f"Error: File not found: {file_path}"
            
            vlog(f"Reading: {target}")
            content = target.read_text(encoding="utf-8", errors="replace")
            
            # Truncate if too long
            if len(content) > 10000:
                content = content[:10000] + f"\n\n[... truncated, {len(content)} total chars ...]"
            
            return content
        
        elif tool_name == "search_codebase":
            # Search for files or patterns
            query = arguments.get("query") or arguments.get("pattern") or arguments.get("search")
            path = arguments.get("path") or arguments.get("directory") or "."
            
            if not query:
                return "Error: No search query provided"
            
            search_path = package_path / path
            if not search_path.exists():
                search_path = Path(path)
            
            vlog(f"Searching in {search_path} for: {query}")
            
            results = []
            try:
                for file in search_path.rglob("*"):
                    if file.is_file() and not any(p in str(file) for p in [".git", "node_modules", "__pycache__", "dist"]):
                        # Match filename
                        if query.lower() in file.name.lower():
                            results.append(f"File: {file.relative_to(search_path)}")
                        # Match in content for text files
                        elif file.suffix in [".ts", ".js", ".py", ".md", ".yaml", ".json", ".txt"]:
                            try:
                                content = file.read_text(encoding="utf-8", errors="ignore")
                                if query.lower() in content.lower():
                                    results.append(f"Match in: {file.relative_to(search_path)}")
                            except Exception:
                                pass
                        
                        if len(results) >= 20:
                            break
            except Exception as e:
                return f"Search error: {e}"
            
            if results:
                return f"Found {len(results)} results:\n" + "\n".join(results[:20])
            return f"No results found for: {query}"
        
        elif tool_name == "analyze_project_structure":
            # List project structure
            path = arguments.get("path") or arguments.get("directory") or "."
            target = package_path / path
            if not target.exists():
                target = Path(path)
            
            vlog(f"Analyzing structure: {target}")
            
            structure = []
            try:
                for item in sorted(target.iterdir()):
                    if item.name.startswith("."):
                        continue
                    if item.is_dir():
                        structure.append(f"📁 {item.name}/")
                    else:
                        structure.append(f"📄 {item.name}")
            except Exception as e:
                return f"Error: {e}"
            
            return f"Project structure of {target}:\n" + "\n".join(structure)
        
        elif tool_name == "check_existing_tests":
            # Find existing test files
            path = arguments.get("path") or arguments.get("directory") or "."
            target = package_path / path
            if not target.exists():
                target = Path(path)
            
            vlog(f"Checking for tests in: {target}")
            
            test_files = []
            test_patterns = ["*.test.*", "*.spec.*", "*_test.*", "*_spec.*", "test_*"]
            
            for pattern in test_patterns:
                for file in target.rglob(pattern):
                    if file.is_file():
                        test_files.append(str(file.relative_to(target)))
            
            if test_files:
                return f"Found {len(test_files)} test files:\n" + "\n".join(sorted(set(test_files))[:30])
            return "No test files found"
        
        elif tool_name == "validate_test_code":
            # Basic validation of test code
            code = arguments.get("code") or arguments.get("content") or ""
            if not code:
                return "Error: No code provided to validate"
            
            issues = []
            
            # Basic checks
            if "test(" not in code and "it(" not in code and "describe(" not in code:
                issues.append("Missing test/it/describe blocks")
            if "expect(" not in code and "assert" not in code:
                issues.append("Missing assertions")
            if "await" in code and "async" not in code:
                issues.append("Using await without async")
            
            if issues:
                return f"Validation issues:\n" + "\n".join(f"• {i}" for i in issues)
            return "Code looks valid! ✓"
        
        else:
            # Unknown tool - return schema info
            return f"Tool '{tool_name}' is a COP definition. Arguments received: {json.dumps(arguments)}"
            
    except Exception as e:
        return f"Tool execution error: {e}"


def _extract_scope_constraints(guardrails: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract scope constraints from guardrails."""
    scope = {
        "blocked_topics": [],
        "required_context": [],
        "off_topic_response": None,
        "min_relevance_score": 0.55,  # Default threshold for semantic check
    }
    
    for gr in guardrails:
        data = gr.get("data", {})
        gr_scope = data.get("scope", {})
        
        scope["blocked_topics"].extend(gr_scope.get("blocked_topics", []))
        scope["required_context"].extend(gr_scope.get("required_context", []))
        
        if gr_scope.get("off_topic_response"):
            scope["off_topic_response"] = gr_scope["off_topic_response"]
        
        # Allow customizing the relevance threshold
        if gr_scope.get("min_relevance_score") is not None:
            scope["min_relevance_score"] = gr_scope["min_relevance_score"]
    
    return scope
