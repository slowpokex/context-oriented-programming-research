"""
MCP Client for COP - Connect to MCP servers and use their tools

This module allows COP agents to consume tools from external MCP servers
during interactive sessions.

Configuration in cop.yaml:
    mcp:
      servers:
        - name: filesystem
          command: npx
          args: ["-y", "@modelcontextprotocol/server-filesystem", "./"]
        - name: browser
          command: python
          args: ["-m", "mcp_server_browser"]
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MCPServerConfig:
        return cls(
            name=data.get("name", "unnamed"),
            command=data.get("command", ""),
            args=data.get("args", []),
            env=data.get("env", {}),
            enabled=data.get("enabled", True),
        )


@dataclass 
class MCPTool:
    """A tool from an MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str
    
    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": f"{self.server_name}__{self.name}",
                "description": f"[{self.server_name}] {self.description}",
                "parameters": self.input_schema,
            }
        }


@dataclass
class MCPResource:
    """A resource from an MCP server."""
    uri: str
    name: str
    description: str
    mime_type: str
    server_name: str


class MCPConnection:
    """Connection to a single MCP server via stdio."""
    
    def __init__(self, config: MCPServerConfig, console: Console, verbose: bool = False):
        self.config = config
        self.console = console
        self.verbose = verbose
        self.process: Optional[subprocess.Popen] = None
        self.tools: List[MCPTool] = []
        self.resources: List[MCPResource] = []
        self._request_id = 0
        self._lock = asyncio.Lock()
    
    def _vlog(self, msg: str) -> None:
        if self.verbose:
            self.console.print(f"[dim]MCP [{self.config.name}]: {msg}[/]")
    
    async def connect(self) -> bool:
        """Start the MCP server process and initialize."""
        try:
            self._vlog(f"Starting: {self.config.command} {' '.join(self.config.args)}")
            
            # Merge environment
            import os
            env = os.environ.copy()
            env.update(self.config.env)
            
            self.process = subprocess.Popen(
                [self.config.command] + self.config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=0,
            )
            
            # Initialize the connection
            init_result = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                },
                "clientInfo": {
                    "name": "cop-cli",
                    "version": "0.1.0"
                }
            })
            
            if init_result is None:
                self._vlog("Initialize failed")
                return False
            
            self._vlog(f"Initialized: {init_result.get('serverInfo', {}).get('name', 'unknown')}")
            
            # Send initialized notification
            await self._send_notification("notifications/initialized", {})
            
            # Fetch tools
            await self._fetch_tools()
            
            # Fetch resources
            await self._fetch_resources()
            
            return True
            
        except FileNotFoundError:
            self.console.print(f"[yellow]MCP [{self.config.name}]:[/] Command not found: {self.config.command}")
            return False
        except Exception as e:
            self.console.print(f"[yellow]MCP [{self.config.name}]:[/] Connection failed: {e}")
            if self.verbose:
                import traceback
                self.console.print(f"[dim]{traceback.format_exc()}[/]")
            return False
    
    async def _fetch_tools(self) -> None:
        """Fetch available tools from the server."""
        result = await self._send_request("tools/list", {})
        if result and "tools" in result:
            self.tools = [
                MCPTool(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                    server_name=self.config.name,
                )
                for t in result["tools"]
            ]
            self._vlog(f"Loaded {len(self.tools)} tools")
    
    async def _fetch_resources(self) -> None:
        """Fetch available resources from the server."""
        result = await self._send_request("resources/list", {})
        if result and "resources" in result:
            self.resources = [
                MCPResource(
                    uri=r.get("uri", ""),
                    name=r.get("name", ""),
                    description=r.get("description", ""),
                    mime_type=r.get("mimeType", "text/plain"),
                    server_name=self.config.name,
                )
                for r in result["resources"]
            ]
            self._vlog(f"Loaded {len(self.resources)} resources")
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on this server."""
        self._vlog(f"Calling tool: {name}")
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        return result
    
    async def read_resource(self, uri: str) -> Optional[str]:
        """Read a resource from this server."""
        self._vlog(f"Reading resource: {uri}")
        result = await self._send_request("resources/read", {"uri": uri})
        if result and "contents" in result:
            contents = result["contents"]
            if contents and len(contents) > 0:
                return contents[0].get("text", "")
        return None
    
    async def _send_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a JSON-RPC request and wait for response."""
        if not self.process or self.process.poll() is not None:
            return None
        
        async with self._lock:
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }
            
            try:
                # Write request
                request_line = json.dumps(request) + "\n"
                self.process.stdin.write(request_line.encode())
                self.process.stdin.flush()
                
                # Read response (with timeout)
                loop = asyncio.get_event_loop()
                response_line = await asyncio.wait_for(
                    loop.run_in_executor(None, self.process.stdout.readline),
                    timeout=30.0
                )
                
                if not response_line:
                    return None
                
                response = json.loads(response_line.decode())
                
                if "error" in response:
                    self._vlog(f"Error: {response['error']}")
                    return None
                
                return response.get("result")
                
            except asyncio.TimeoutError:
                self._vlog(f"Request timeout: {method}")
                return None
            except Exception as e:
                self._vlog(f"Request error: {e}")
                return None
    
    async def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self.process or self.process.poll() is not None:
            return
        
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        
        try:
            notification_line = json.dumps(notification) + "\n"
            self.process.stdin.write(notification_line.encode())
            self.process.stdin.flush()
        except Exception:
            pass
    
    def disconnect(self) -> None:
        """Terminate the MCP server process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None
            self._vlog("Disconnected")


class MCPManager:
    """Manages multiple MCP server connections."""
    
    def __init__(self, console: Console, verbose: bool = False):
        self.console = console
        self.verbose = verbose
        self.connections: Dict[str, MCPConnection] = {}
    
    async def connect_servers(self, configs: List[MCPServerConfig]) -> int:
        """Connect to multiple MCP servers. Returns count of successful connections."""
        connected = 0
        
        for config in configs:
            if not config.enabled:
                continue
            
            conn = MCPConnection(config, self.console, self.verbose)
            if await conn.connect():
                self.connections[config.name] = conn
                connected += 1
        
        return connected
    
    def get_all_tools(self) -> List[MCPTool]:
        """Get all tools from all connected servers."""
        tools = []
        for conn in self.connections.values():
            tools.extend(conn.tools)
        return tools
    
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI function calling format."""
        return [tool.to_openai_tool() for tool in self.get_all_tools()]
    
    async def call_tool(self, full_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by its full name (server__tool)."""
        if "__" not in full_name:
            return {"error": f"Invalid tool name format: {full_name}"}
        
        server_name, tool_name = full_name.split("__", 1)
        
        if server_name not in self.connections:
            return {"error": f"Unknown MCP server: {server_name}"}
        
        conn = self.connections[server_name]
        return await conn.call_tool(tool_name, arguments)
    
    def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for conn in self.connections.values():
            conn.disconnect()
        self.connections.clear()


def load_mcp_config(manifest: Dict[str, Any]) -> List[MCPServerConfig]:
    """Load MCP configuration from cop.yaml manifest."""
    mcp_config = manifest.get("mcp", {})
    servers = mcp_config.get("servers", [])
    
    return [MCPServerConfig.from_dict(s) for s in servers]


async def setup_mcp_tools(
    manifest: Dict[str, Any],
    console: Console,
    verbose: bool = False
) -> Optional[MCPManager]:
    """Setup MCP connections from manifest config.
    
    Returns MCPManager if any servers connected, None otherwise.
    """
    configs = load_mcp_config(manifest)
    
    if not configs:
        return None
    
    if verbose:
        console.print(f"[dim]MCP: Found {len(configs)} server configs[/]")
    
    manager = MCPManager(console, verbose)
    connected = await manager.connect_servers(configs)
    
    if connected == 0:
        console.print("[dim]MCP: No servers connected[/]")
        return None
    
    console.print(f"[dim]MCP: Connected to {connected} server(s), {len(manager.get_all_tools())} tools available[/]")
    
    return manager

