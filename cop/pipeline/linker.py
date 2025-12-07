"""
Data Linker for COP Packages

Orchestrates the full data linking pipeline:
1. Resolve dependencies (import tree)
2. Chunk content
3. Generate embeddings
4. Build vector index
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import APIError, APIConnectionError, RateLimitError
from rich.console import Console

from .resolver import DependencyResolver, ImportTree, ContentType
from .chunker import ContentChunker, ChunkerConfig, ChunkingStrategy, Chunk
from .embeddings import EmbeddingGenerator, EmbeddingConfig, EmbeddingBatch
from .indexer import VectorIndex, IndexBuilder, IndexFormat, IndexStats
from .constants import (
    DEFAULT_EMBEDDING_ENDPOINT, DEFAULT_EMBEDDING_MODEL, DEFAULT_API_KEY,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_MIN_CHUNK_SIZE,
    DEFAULT_CHUNKING_STRATEGY, FAISS_AVAILABLE
)


@dataclass
class LinkingConfig:
    """Configuration for the data linking pipeline.
    
    Most settings have sensible defaults. Minimal config:
    
        linking:
          enabled: true
    
    Everything else is auto-detected or uses environment variables.
    """
    # Embedding settings (from env vars or cop.yaml)
    embedding_endpoint: str = DEFAULT_EMBEDDING_ENDPOINT
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    api_key: str = DEFAULT_API_KEY
    
    # Chunking settings (sensible defaults)
    chunking_strategy: ChunkingStrategy = None  # Auto-set in __post_init__
    max_chunk_size: int = DEFAULT_CHUNK_SIZE
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    
    # Index settings (auto-detect FAISS)
    index_format: IndexFormat = None  # Auto-set in __post_init__
    use_faiss: bool = None  # Auto-detect
    
    # Cache settings
    cache_embeddings: bool = True
    cache_dir: Optional[Path] = None
    
    # What to include (all by default except tests)
    include_prompts: bool = True
    include_personas: bool = True
    include_guardrails: bool = True
    include_knowledge: bool = True
    include_tools: bool = True
    include_tests: bool = False
    
    def __post_init__(self):
        """Auto-detect settings that weren't explicitly set."""
        # Auto-detect FAISS availability
        if self.use_faiss is None:
            self.use_faiss = FAISS_AVAILABLE
        
        # Auto-set index format based on FAISS
        if self.index_format is None:
            self.index_format = IndexFormat.FAISS if self.use_faiss else IndexFormat.JSON
        
        # Default chunking strategy
        if self.chunking_strategy is None:
            strategy_str = DEFAULT_CHUNKING_STRATEGY
            self.chunking_strategy = (
                ChunkingStrategy(strategy_str) 
                if strategy_str in [s.value for s in ChunkingStrategy] 
                else ChunkingStrategy.SEMANTIC
            )
    
    @classmethod
    def from_manifest(cls, build_config: Dict[str, Any]) -> 'LinkingConfig':
        """Create config from cop.yaml build section.
        
        Supports both verbose and minimal manifest formats:
        
        Minimal:
            linking:
              enabled: true
        
        Verbose (optional overrides):
            linking:
              enabled: true
              embedding_model: "custom-model"
              chunking:
                strategy: "semantic"
                max_chunk_size: 512
        """
        linking = build_config.get("linking", {})
        local_llm = build_config.get("local_llm", {})
        
        # Chunking settings (use defaults if not specified)
        chunking = linking.get("chunking", {})
        strategy_str = chunking.get("strategy")
        strategy = None
        if strategy_str:
            strategy = (
                ChunkingStrategy(strategy_str) 
                if strategy_str in [s.value for s in ChunkingStrategy] 
                else None
            )
        
        # Index settings (auto-detect if not specified)
        index = linking.get("index", {})
        format_str = index.get("format")
        index_format = None
        if format_str:
            index_format = (
                IndexFormat(format_str) 
                if format_str in [f.value for f in IndexFormat] 
                else None
            )
        
        return cls(
            embedding_endpoint=local_llm.get("endpoint", DEFAULT_EMBEDDING_ENDPOINT),
            embedding_model=linking.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            api_key=local_llm.get("api_key", DEFAULT_API_KEY),
            chunking_strategy=strategy,
            max_chunk_size=chunking.get("max_chunk_size", DEFAULT_CHUNK_SIZE),
            min_chunk_size=chunking.get("min_chunk_size", DEFAULT_MIN_CHUNK_SIZE),
            chunk_overlap=chunking.get("overlap", DEFAULT_CHUNK_OVERLAP),
            index_format=index_format,
            use_faiss=None,  # Auto-detect
            cache_embeddings=linking.get("cache", True),
            include_prompts=True,
            include_personas=True,
            include_guardrails=True,
            include_knowledge=True,
            include_tools=True,
            include_tests=linking.get("include_tests", False)
        )


@dataclass
class LinkingResult:
    """Result of the data linking pipeline."""
    import_tree: ImportTree
    chunks: List[Chunk]
    embeddings: Optional[EmbeddingBatch]
    index: Optional[VectorIndex]
    stats: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_files(self) -> int:
        return self.import_tree.total_files
    
    @property
    def total_chunks(self) -> int:
        return len(self.chunks)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_files": self.total_files,
            "total_chunks": self.total_chunks,
            "files_by_type": self.import_tree.summary(),
            "embedding_dimensions": self.embeddings.dimensions if self.embeddings else 0,
            "processing_time_ms": self.stats.get("total_time_ms", 0)
        }


class DataLinker:
    """
    Orchestrates the full data linking pipeline for a COP package.
    """
    
    def __init__(
        self,
        package_path: Path,
        config: Optional[LinkingConfig] = None,
        console: Optional[Console] = None,
        verbose: bool = False
    ):
        self.package_path = Path(package_path)
        self.config = config or LinkingConfig()
        self.console = console or Console()
        self.verbose = verbose
    
    def link(self, skip_embeddings: bool = False) -> LinkingResult:
        """
        Run the full data linking pipeline.
        
        Args:
            skip_embeddings: If True, only resolve and chunk (no embeddings)
            
        Returns:
            LinkingResult with all artifacts
        """
        start_time = time.time()
        stats = {}
        
        # Step 1: Resolve dependencies
        if self.verbose:
            self.console.print("[bold blue]Step 1:[/] Resolving dependencies...")
        
        resolver = DependencyResolver(self.package_path)
        import_tree = resolver.resolve()
        
        stats["resolve_time_ms"] = (time.time() - start_time) * 1000
        
        if self.verbose:
            self.console.print(f"  [dim]Found {import_tree.total_files} files[/]")
            for content_type, count in import_tree.summary().items():
                self.console.print(f"    - {content_type}: {count}")
        
        # Step 2: Filter and chunk content
        step2_start = time.time()
        if self.verbose:
            self.console.print("[bold blue]Step 2:[/] Chunking content...")
        
        files_to_chunk = self._filter_files(import_tree)
        
        chunker_config = ChunkerConfig(
            strategy=self.config.chunking_strategy,
            max_chunk_size=self.config.max_chunk_size,
            min_chunk_size=self.config.min_chunk_size,
            overlap=self.config.chunk_overlap
        )
        chunker = ContentChunker(chunker_config)
        chunks = chunker.chunk_files(files_to_chunk)
        
        stats["chunk_time_ms"] = (time.time() - step2_start) * 1000
        total_chars = sum(len(c.content) for c in chunks)
        
        if self.verbose:
            self.console.print(f"  [dim]Created {len(chunks)} chunks ({total_chars:,} chars total)[/]")
            # Group chunks by content type
            by_type: Dict[str, int] = {}
            for chunk in chunks:
                type_name = chunk.content_type.value
                by_type[type_name] = by_type.get(type_name, 0) + 1
            for type_name, count in by_type.items():
                self.console.print(f"    - {type_name}: {count} chunks")
        
        # Step 3: Generate embeddings (optional)
        embeddings = None
        index = None
        
        if not skip_embeddings:
            step3_start = time.time()
            if self.verbose:
                self.console.print("[bold blue]Step 3:[/] Generating embeddings...")
            
            embed_config = EmbeddingConfig(
                endpoint=self.config.embedding_endpoint,
                model=self.config.embedding_model,
                api_key=self.config.api_key,
                cache_dir=self.config.cache_dir if self.config.cache_embeddings else None
            )
            generator = EmbeddingGenerator(embed_config, verbose=self.verbose)
            
            try:
                embeddings = generator.embed_chunks(
                    chunks,
                    show_progress=self.verbose,
                    console=self.console
                )
                stats["embed_time_ms"] = (time.time() - step3_start) * 1000
                
                if self.verbose:
                    self.console.print(f"  [dim]Generated {len(embeddings.embeddings)} embeddings[/]")
                    self.console.print(f"  [dim]Dimensions: {embeddings.dimensions}[/]")
                
                # Step 4: Build index
                step4_start = time.time()
                if self.verbose:
                    self.console.print("[bold blue]Step 4:[/] Building vector index...")
                
                builder = IndexBuilder(use_faiss=self.config.use_faiss)
                index = builder.build(embeddings, chunks)
                
                stats["index_time_ms"] = (time.time() - step4_start) * 1000
                
                if self.verbose:
                    self.console.print(f"  [dim]Index ready with {len(index.chunk_ids)} vectors[/]")
                    
            except (APIError, APIConnectionError, RateLimitError) as e:
                if self.verbose:
                    self.console.print(f"  Warning: Embedding API failed: {e}", style="yellow", markup=False)
                    self.console.print("  Continuing without embeddings...", style="dim")
            except (ValueError, TypeError) as e:
                if self.verbose:
                    self.console.print(f"  Warning: Embedding data error: {e}", style="yellow", markup=False)
                    self.console.print("  Continuing without embeddings...", style="dim")
        
        stats["total_time_ms"] = (time.time() - start_time) * 1000
        
        return LinkingResult(
            import_tree=import_tree,
            chunks=chunks,
            embeddings=embeddings,
            index=index,
            stats=stats
        )
    
    def _filter_files(self, import_tree: ImportTree) -> List:
        """Filter files based on config."""
        from .resolver import ResolvedFile
        
        files: List[ResolvedFile] = []
        
        type_config = {
            ContentType.PROMPT: self.config.include_prompts,
            ContentType.PERSONA: self.config.include_personas,
            ContentType.GUARDRAIL: self.config.include_guardrails,
            ContentType.KNOWLEDGE: self.config.include_knowledge,
            ContentType.TOOL: self.config.include_tools,
            ContentType.TEST: self.config.include_tests,
        }
        
        for resolved_file in import_tree.files.values():
            include = type_config.get(resolved_file.content_type, True)
            if include:
                files.append(resolved_file)
        
        return files
    
    def save_artifacts(
        self,
        result: LinkingResult,
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Save linking artifacts to disk.
        
        Args:
            result: LinkingResult from link()
            output_dir: Directory to save artifacts
            
        Returns:
            Dict mapping artifact names to paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        
        import json
        
        # Save chunks
        chunks_file = output_dir / "chunks.jsonl"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for chunk in result.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')
        artifacts["chunks"] = chunks_file
        
        # Save index if available
        if result.index is not None:
            index_stats = result.index.save(output_dir, self.config.index_format)
            artifacts["index"] = output_dir / "index_metadata.json"
            
            if self.verbose:
                self.console.print(f"  [dim]Index saved: {index_stats.index_size_bytes / 1024:.1f} KB[/]")
        
        # Save linking metadata
        metadata = {
            "package": str(self.package_path),
            "stats": result.stats,
            "summary": result.summary()
        }
        metadata_file = output_dir / "linking_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        artifacts["metadata"] = metadata_file
        
        return artifacts


def link_package(
    package_path: str | Path,
    config: Optional[LinkingConfig] = None,
    output_dir: Optional[Path] = None,
    console: Optional[Console] = None,
    verbose: bool = False,
    skip_embeddings: bool = False
) -> LinkingResult:
    """
    Convenience function to link a COP package.
    
    Args:
        package_path: Path to COP package
        config: Optional linking configuration
        output_dir: Optional directory to save artifacts
        console: Rich console for output
        verbose: Whether to show progress
        skip_embeddings: Skip embedding generation
        
    Returns:
        LinkingResult with all artifacts
    """
    linker = DataLinker(
        package_path=Path(package_path),
        config=config,
        console=console,
        verbose=verbose
    )
    
    result = linker.link(skip_embeddings=skip_embeddings)
    
    if output_dir:
        linker.save_artifacts(result, Path(output_dir))
    
    return result

