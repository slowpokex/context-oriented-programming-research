"""
Enhanced Logging Utilities for COP Pipeline

Provides detailed verbose logging for:
- API calls (endpoint, model, tokens)
- Timing metrics
- Connectivity status
- Progress indicators
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


@dataclass
class APICallStats:
    """Statistics for an API call."""
    endpoint: str
    model: str
    operation: str  # "chat", "embedding", etc.
    request_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "operation": self.operation,
            "request_tokens": self.request_tokens,
            "response_tokens": self.response_tokens,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PipelineStats:
    """Aggregated statistics for a pipeline run."""
    api_calls: List[APICallStats] = field(default_factory=list)
    total_api_calls: int = 0
    total_tokens_sent: int = 0
    total_tokens_received: int = 0
    total_api_time_ms: float = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_call(self, stats: APICallStats) -> None:
        self.api_calls.append(stats)
        self.total_api_calls += 1
        self.total_tokens_sent += stats.request_tokens
        self.total_tokens_received += stats.response_tokens
        self.total_api_time_ms += stats.duration_ms
        if stats.error:
            self.errors.append(stats.error)
    
    def summary(self) -> Dict[str, Any]:
        return {
            "total_api_calls": self.total_api_calls,
            "total_tokens_sent": self.total_tokens_sent,
            "total_tokens_received": self.total_tokens_received,
            "total_tokens": self.total_tokens_sent + self.total_tokens_received,
            "total_api_time_ms": self.total_api_time_ms,
            "avg_call_time_ms": self.total_api_time_ms / max(1, self.total_api_calls),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }


class PipelineLogger:
    """
    Enhanced logger for verbose pipeline output.
    
    Provides detailed logging of:
    - API calls with token counts
    - Timing for operations
    - Connectivity status
    - Progress indicators
    """
    
    def __init__(
        self,
        console: Optional[Console] = None,
        verbose: bool = False,
        collect_stats: bool = True
    ):
        self.console = console or Console()
        self.verbose = verbose
        self.collect_stats = collect_stats
        self.stats = PipelineStats()
        self._indent_level = 0
    
    def _prefix(self) -> str:
        """Get indentation prefix."""
        return "  " * self._indent_level
    
    def _log(self, message: str, style: str = "dim") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            self.console.print(f"{self._prefix()}[{style}]{message}[/]")
    
    def info(self, message: str) -> None:
        """Log info message."""
        self._log(message, "dim")
    
    def success(self, message: str) -> None:
        """Log success message."""
        self._log(message, "green")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self._log(f"⚠ {message}", "yellow")
        if self.collect_stats:
            self.stats.warnings.append(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self._log(f"✗ {message}", "red")
        if self.collect_stats:
            self.stats.errors.append(message)
    
    def section(self, title: str) -> None:
        """Log a section header."""
        if self.verbose:
            self.console.print(f"{self._prefix()}[bold cyan]▸ {title}[/]")
    
    @contextmanager
    def indent(self):
        """Context manager for indented logging."""
        self._indent_level += 1
        try:
            yield
        finally:
            self._indent_level -= 1
    
    @contextmanager
    def timed_operation(self, operation_name: str):
        """Context manager that times an operation."""
        start = time.time()
        self.info(f"{operation_name}...")
        try:
            yield
        finally:
            duration = (time.time() - start) * 1000
            self.info(f"{operation_name} completed in {duration:.1f}ms")
    
    def log_api_call_start(
        self,
        endpoint: str,
        model: str,
        operation: str,
        input_chars: int = 0
    ) -> float:
        """Log the start of an API call, return start time."""
        estimated_tokens = input_chars // 4  # Rough estimate
        self.info(f"API → {operation} | model={model}")
        self.info(f"     endpoint={endpoint}")
        self.info(f"     input: ~{estimated_tokens} tokens ({input_chars} chars)")
        return time.time()
    
    def log_api_call_end(
        self,
        start_time: float,
        endpoint: str,
        model: str,
        operation: str,
        request_tokens: int = 0,
        response_tokens: int = 0,
        total_tokens: int = 0,
        success: bool = True,
        error: Optional[str] = None
    ) -> APICallStats:
        """Log the end of an API call."""
        duration_ms = (time.time() - start_time) * 1000
        
        stats = APICallStats(
            endpoint=endpoint,
            model=model,
            operation=operation,
            request_tokens=request_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens or (request_tokens + response_tokens),
            duration_ms=duration_ms,
            success=success,
            error=error
        )
        
        if self.collect_stats:
            self.stats.add_call(stats)
        
        if success:
            self.info(f"API ← {operation} | {duration_ms:.0f}ms | tokens: {request_tokens}→{response_tokens} (total: {stats.total_tokens})")
        else:
            self.error(f"API ✗ {operation} | {duration_ms:.0f}ms | error: {error}")
        
        return stats
    
    def log_connectivity_error(
        self,
        endpoint: str,
        error_type: str,
        message: str,
        retry_count: int = 0,
        max_retries: int = 0
    ) -> None:
        """Log a connectivity error."""
        if retry_count > 0:
            self.warning(f"Connection error ({error_type}): {message}")
            self.info(f"     Retry {retry_count}/{max_retries}...")
        else:
            self.error(f"Connection failed ({error_type}): {message}")
            self.error(f"     endpoint: {endpoint}")
    
    def log_embedding_batch(
        self,
        batch_num: int,
        total_batches: int,
        batch_size: int,
        dimensions: int,
        duration_ms: float
    ) -> None:
        """Log embedding batch progress."""
        self.info(f"Embedding batch {batch_num}/{total_batches} | {batch_size} chunks | {dimensions}d | {duration_ms:.0f}ms")
    
    def log_file_operation(
        self,
        operation: str,
        file_path: str,
        size_bytes: int = 0,
        duration_ms: float = 0
    ) -> None:
        """Log file operation."""
        size_str = f"{size_bytes / 1024:.1f}KB" if size_bytes > 0 else ""
        time_str = f" | {duration_ms:.0f}ms" if duration_ms > 0 else ""
        self.info(f"File {operation}: {file_path} {size_str}{time_str}")
    
    def log_chunk_stats(
        self,
        total_files: int,
        total_chunks: int,
        total_chars: int,
        by_type: Dict[str, int] = None
    ) -> None:
        """Log chunking statistics."""
        self.info(f"Chunking complete: {total_files} files → {total_chunks} chunks ({total_chars:,} chars)")
        if by_type:
            for content_type, count in by_type.items():
                self.info(f"     {content_type}: {count} chunks")
    
    def log_rag_retrieval(
        self,
        query_preview: str,
        results_count: int,
        top_score: float,
        duration_ms: float
    ) -> None:
        """Log RAG retrieval."""
        query_short = query_preview[:50] + "..." if len(query_preview) > 50 else query_preview
        self.info(f"RAG query: \"{query_short}\"")
        self.info(f"     results: {results_count} | top_score: {top_score:.3f} | {duration_ms:.0f}ms")
    
    def print_stats_summary(self) -> None:
        """Print a summary table of collected statistics."""
        if not self.verbose or not self.collect_stats:
            return
        
        summary = self.stats.summary()
        
        self.console.print()
        table = Table(title="Pipeline Statistics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        
        table.add_row("Total API Calls", str(summary["total_api_calls"]))
        table.add_row("Tokens Sent", f"{summary['total_tokens_sent']:,}")
        table.add_row("Tokens Received", f"{summary['total_tokens_received']:,}")
        table.add_row("Total Tokens", f"{summary['total_tokens']:,}")
        table.add_row("Total API Time", f"{summary['total_api_time_ms']:.0f}ms")
        table.add_row("Avg Call Time", f"{summary['avg_call_time_ms']:.0f}ms")
        table.add_row("Errors", str(summary["error_count"]))
        table.add_row("Warnings", str(summary["warning_count"]))
        
        self.console.print(table)

