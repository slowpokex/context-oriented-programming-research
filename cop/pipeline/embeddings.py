"""
Embedding Generator for COP Packages

Generates embeddings for content chunks using local LLM (LM Studio/Ollama)
or cloud embedding APIs.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError

from .chunker import Chunk
from .logger import PipelineLogger
from .constants import DEFAULT_EMBEDDING_ENDPOINT, DEFAULT_EMBEDDING_MODEL, DEFAULT_API_KEY


@dataclass
class EmbeddingResult:
    """Result of embedding a chunk."""
    chunk_id: str
    embedding: List[float]
    model: str
    dimensions: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "embedding": self.embedding,
            "model": self.model,
            "dimensions": self.dimensions
        }


@dataclass
class EmbeddingBatch:
    """A batch of embeddings with metadata."""
    embeddings: List[EmbeddingResult]
    model: str
    dimensions: int
    total_chunks: int
    processing_time_ms: float
    
    def to_numpy(self) -> np.ndarray:
        """Convert embeddings to numpy array."""
        return np.array([e.embedding for e in self.embeddings], dtype=np.float32)
    
    def get_chunk_ids(self) -> List[str]:
        """Get list of chunk IDs in order."""
        return [e.chunk_id for e in self.embeddings]


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    endpoint: str = DEFAULT_EMBEDDING_ENDPOINT
    model: str = DEFAULT_EMBEDDING_MODEL
    api_key: str = DEFAULT_API_KEY
    batch_size: int = 32           # Chunks per API call
    max_retries: int = 3
    retry_delay_ms: int = 1000
    normalize: bool = True         # L2 normalize embeddings
    cache_dir: Optional[Path] = None  # Cache embeddings to disk


class EmbeddingGenerator:
    """
    Generates embeddings for content chunks using local or cloud models.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None, verbose: bool = False):
        self.config = config or EmbeddingConfig()
        self.verbose = verbose
        self._client: Optional[OpenAI] = None
        self._cache: Dict[str, List[float]] = {}
        self._dimensions: Optional[int] = None
        self._logger: Optional[PipelineLogger] = None
        self._batch_count: int = 0
        
    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.config.endpoint,
                api_key=self.config.api_key
            )
        return self._client
    
    def _get_logger(self, console: Optional[Any] = None) -> PipelineLogger:
        """Get or create logger."""
        if self._logger is None:
            from rich.console import Console
            self._logger = PipelineLogger(
                console=console or Console(),
                verbose=self.verbose,
                collect_stats=True
            )
        return self._logger
    
    def embed_chunks(
        self,
        chunks: List[Chunk],
        show_progress: bool = True,
        console: Optional[Any] = None
    ) -> EmbeddingBatch:
        """
        Generate embeddings for all chunks.
        
        Args:
            chunks: List of chunks to embed
            show_progress: Whether to show progress
            console: Rich console for output
            
        Returns:
            EmbeddingBatch with all embeddings
        """
        logger = self._get_logger(console)
        start_time = time.time()
        embeddings: List[EmbeddingResult] = []
        total_tokens_used = 0
        
        if self.verbose:
            logger.section("Embedding Generation")
            with logger.indent():
                logger.info(f"Endpoint: {self.config.endpoint}")
                logger.info(f"Model: {self.config.model}")
                logger.info(f"Batch size: {self.config.batch_size}")
                logger.info(f"Total chunks: {len(chunks)}")
        
        # Load cache if configured
        self._load_cache()
        cached_count = 0
        
        # Filter out cached chunks
        uncached_chunks = []
        for chunk in chunks:
            cache_key = self._cache_key(chunk)
            if cache_key in self._cache:
                embeddings.append(EmbeddingResult(
                    chunk_id=chunk.id,
                    embedding=self._cache[cache_key],
                    model=self.config.model,
                    dimensions=len(self._cache[cache_key])
                ))
                cached_count += 1
            else:
                uncached_chunks.append(chunk)
        
        if self.verbose:
            logger.info(f"Cache hit: {cached_count}/{len(chunks)} chunks")
            logger.info(f"To embed: {len(uncached_chunks)} chunks")
        
        # Batch process uncached chunks
        total_batches = (len(uncached_chunks) + self.config.batch_size - 1) // self.config.batch_size
        
        for i in range(0, len(uncached_chunks), self.config.batch_size):
            self._batch_count += 1
            batch = uncached_chunks[i:i + self.config.batch_size]
            batch_start = time.time()
            
            batch_results = self._embed_batch(batch, logger)
            embeddings.extend(batch_results)
            
            batch_duration = (time.time() - batch_start) * 1000
            batch_num = (i // self.config.batch_size) + 1
            dimensions = batch_results[0].dimensions if batch_results else 0
            
            if self.verbose:
                logger.log_embedding_batch(
                    batch_num=batch_num,
                    total_batches=total_batches,
                    batch_size=len(batch),
                    dimensions=dimensions,
                    duration_ms=batch_duration
                )
        
        # Save cache
        self._save_cache()
        
        processing_time = (time.time() - start_time) * 1000
        
        # Determine dimensions
        dimensions = embeddings[0].dimensions if embeddings else 0
        
        if self.verbose:
            logger.success(f"Embedding complete: {len(embeddings)} vectors, {dimensions}d, {processing_time:.0f}ms total")
        
        return EmbeddingBatch(
            embeddings=embeddings,
            model=self.config.model,
            dimensions=dimensions,
            total_chunks=len(chunks),
            processing_time_ms=processing_time
        )
    
    def embed_single(self, text: str) -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            response = self.client.embeddings.create(
                model=self.config.model,
                input=text
            )
            embedding = response.data[0].embedding
            
            if self.config.normalize:
                embedding = self._normalize(embedding)
            
            self._cache[cache_key] = embedding
            return embedding
            
        except (APIError, APIConnectionError, RateLimitError) as e:
            raise RuntimeError(f"Embedding API error: {e}")
    
    def _embed_batch(self, chunks: List[Chunk], logger: Optional[PipelineLogger] = None) -> List[EmbeddingResult]:
        """Embed a batch of chunks."""
        if not chunks:
            return []
        
        texts = [chunk.content for chunk in chunks]
        total_chars = sum(len(t) for t in texts)
        
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                
                if logger and self.verbose:
                    logger.log_api_call_start(
                        endpoint=self.config.endpoint,
                        model=self.config.model,
                        operation=f"embedding (batch {self._batch_count})",
                        input_chars=total_chars
                    )
                
                response = self.client.embeddings.create(
                    model=self.config.model,
                    input=texts
                )
                
                duration_ms = (time.time() - start_time) * 1000
                
                # Extract token usage if available
                usage = response.usage if hasattr(response, 'usage') and response.usage else None
                tokens_used = usage.total_tokens if usage else total_chars // 4
                
                if logger and self.verbose:
                    logger.log_api_call_end(
                        start_time=start_time,
                        endpoint=self.config.endpoint,
                        model=self.config.model,
                        operation=f"embedding (batch {self._batch_count})",
                        request_tokens=tokens_used,
                        response_tokens=0,  # Embeddings don't have response tokens
                        total_tokens=tokens_used,
                        success=True
                    )
                
                results = []
                for i, data in enumerate(response.data):
                    embedding = data.embedding
                    
                    if self.config.normalize:
                        embedding = self._normalize(embedding)
                    
                    # Cache the result
                    cache_key = self._cache_key(chunks[i])
                    self._cache[cache_key] = embedding
                    
                    results.append(EmbeddingResult(
                        chunk_id=chunks[i].id,
                        embedding=embedding,
                        model=self.config.model,
                        dimensions=len(embedding)
                    ))
                
                return results
                
            except RateLimitError as e:
                if logger and self.verbose:
                    logger.log_connectivity_error(
                        endpoint=self.config.endpoint,
                        error_type="RateLimitError",
                        message="Rate limited by embedding API",
                        retry_count=attempt + 1,
                        max_retries=self.config.max_retries
                    )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_ms / 1000)
                    continue
                raise
            except APIConnectionError as e:
                if logger and self.verbose:
                    logger.log_connectivity_error(
                        endpoint=self.config.endpoint,
                        error_type="APIConnectionError",
                        message=str(e)[:100],
                        retry_count=attempt + 1,
                        max_retries=self.config.max_retries
                    )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_ms / 1000)
                    continue
                raise RuntimeError(f"Failed to connect to embedding API: {e}")
            except AuthenticationError as e:
                if logger and self.verbose:
                    logger.log_connectivity_error(
                        endpoint=self.config.endpoint,
                        error_type="AuthenticationError",
                        message="Invalid API key"
                    )
                raise RuntimeError(f"Embedding API authentication failed: {e}")
            except APIError as e:
                if logger and self.verbose:
                    logger.log_connectivity_error(
                        endpoint=self.config.endpoint,
                        error_type="APIError",
                        message=str(e)[:100]
                    )
                raise RuntimeError(f"Embedding API error: {e}")
        
        return []
    
    def _normalize(self, embedding: List[float]) -> List[float]:
        """L2 normalize an embedding vector."""
        arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()
    
    def _cache_key(self, chunk: Chunk) -> str:
        """Generate cache key for a chunk."""
        return hashlib.sha256(chunk.content.encode()).hexdigest()
    
    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        if self.config.cache_dir is None:
            return
        
        cache_file = self.config.cache_dir / "embedding_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError, OSError) as e:
                # Cache is optional - log and continue with empty cache
                if self.verbose and self._logger:
                    self._logger.warning(f"Cache load failed: {e}")
                self._cache = {}
    
    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        if self.config.cache_dir is None:
            return
        
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.config.cache_dir / "embedding_cache.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f)
        except (IOError, OSError, TypeError) as e:
            # Cache is optional - log but don't fail the pipeline
            if self.verbose and self._logger:
                self._logger.warning(f"Cache save failed: {e}")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot / (norm_a * norm_b))


def embed_chunks(
    chunks: List[Chunk],
    endpoint: str = DEFAULT_EMBEDDING_ENDPOINT,
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str = DEFAULT_API_KEY,
    console: Optional[Any] = None
) -> EmbeddingBatch:
    """
    Convenience function to embed chunks.
    
    Args:
        chunks: Chunks to embed
        endpoint: API endpoint
        model: Embedding model name
        api_key: API key
        console: Rich console for progress
        
    Returns:
        EmbeddingBatch with all embeddings
    """
    config = EmbeddingConfig(
        endpoint=endpoint,
        model=model,
        api_key=api_key
    )
    generator = EmbeddingGenerator(config)
    return generator.embed_chunks(chunks, console=console)

