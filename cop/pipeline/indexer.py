"""
Vector Indexer for COP Packages

Builds searchable vector indexes from embeddings using FAISS or JSON fallback.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

import numpy as np

from .chunker import Chunk
from .embeddings import EmbeddingBatch, EmbeddingResult, cosine_similarity

# Check FAISS availability once at module load (not per-instance)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None  # type: ignore


class IndexFormat(Enum):
    """Available index formats."""
    FAISS = "faiss"
    JSON = "json"      # Fallback when FAISS not available
    NUMPY = "numpy"    # NumPy binary format


@dataclass
class SearchResult:
    """A single search result."""
    chunk_id: str
    score: float
    chunk: Optional[Chunk] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "chunk": self.chunk.to_dict() if self.chunk else None
        }


@dataclass
class IndexStats:
    """Statistics about the vector index."""
    total_vectors: int
    dimensions: int
    index_format: str
    index_size_bytes: int
    build_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_vectors": self.total_vectors,
            "dimensions": self.dimensions,
            "index_format": self.index_format,
            "index_size_bytes": self.index_size_bytes,
            "build_time_ms": self.build_time_ms
        }


class VectorIndex:
    """
    A searchable vector index supporting FAISS or JSON fallback.
    """
    
    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        chunk_ids: Optional[List[str]] = None,
        chunks: Optional[Dict[str, Chunk]] = None,
        dimensions: int = 0,
        use_faiss: bool = True
    ):
        self.embeddings = embeddings
        self.chunk_ids = chunk_ids or []
        self.chunks = chunks or {}
        self.dimensions = dimensions
        self._faiss_index = None
        self._use_faiss = use_faiss and FAISS_AVAILABLE
        
        if embeddings is not None and self._use_faiss:
            self._build_faiss_index()
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index from embeddings."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return
        
        if not FAISS_AVAILABLE or faiss is None:
            self._use_faiss = False
            return
        
        # Use inner product index (works well with normalized vectors for cosine sim)
        self._faiss_index = faiss.IndexFlatIP(self.dimensions)
        self._faiss_index.add(self.embeddings.astype(np.float32))
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search the index for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects sorted by score
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        query = np.array(query_embedding, dtype=np.float32)
        
        if self._use_faiss and self._faiss_index is not None:
            return self._search_faiss(query, top_k, min_score)
        else:
            return self._search_brute_force(query, top_k, min_score)
    
    def _search_faiss(
        self,
        query: np.ndarray,
        top_k: int,
        min_score: float
    ) -> List[SearchResult]:
        """Search using FAISS."""
        query = query.reshape(1, -1)
        scores, indices = self._faiss_index.search(query, min(top_k, len(self.chunk_ids)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= min_score:
                chunk_id = self.chunk_ids[idx]
                results.append(SearchResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    chunk=self.chunks.get(chunk_id)
                ))
        
        return results
    
    def _search_brute_force(
        self,
        query: np.ndarray,
        top_k: int,
        min_score: float
    ) -> List[SearchResult]:
        """Brute force search (fallback when FAISS not available)."""
        scores = []
        
        for i, embedding in enumerate(self.embeddings):
            score = cosine_similarity(query.tolist(), embedding.tolist())
            if score >= min_score:
                scores.append((score, i))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, idx in scores[:top_k]:
            chunk_id = self.chunk_ids[idx]
            results.append(SearchResult(
                chunk_id=chunk_id,
                score=float(score),
                chunk=self.chunks.get(chunk_id)
            ))
        
        return results
    
    def save(self, output_dir: Path, format: IndexFormat = IndexFormat.JSON) -> IndexStats:
        """
        Save the index to disk.
        
        Args:
            output_dir: Directory to save index files
            format: Output format
            
        Returns:
            IndexStats about the saved index
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        total_size = 0
        
        if format == IndexFormat.FAISS and self._use_faiss and self._faiss_index:
            total_size = self._save_faiss(output_dir)
        elif format == IndexFormat.NUMPY:
            total_size = self._save_numpy(output_dir)
        else:
            total_size = self._save_json(output_dir)
        
        # Always save chunks metadata
        chunks_file = output_dir / "chunks.jsonl"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for chunk_id in self.chunk_ids:
                if chunk_id in self.chunks:
                    chunk_data = self.chunks[chunk_id].to_dict()
                    f.write(json.dumps(chunk_data) + '\n')
        total_size += chunks_file.stat().st_size
        
        # Save index metadata
        metadata = {
            "format": format.value,
            "dimensions": self.dimensions,
            "total_vectors": len(self.chunk_ids),
            "chunk_ids": self.chunk_ids
        }
        metadata_file = output_dir / "index_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        total_size += metadata_file.stat().st_size
        
        build_time = (time.time() - start_time) * 1000
        
        return IndexStats(
            total_vectors=len(self.chunk_ids),
            dimensions=self.dimensions,
            index_format=format.value,
            index_size_bytes=total_size,
            build_time_ms=build_time
        )
    
    def _save_faiss(self, output_dir: Path) -> int:
        """Save as FAISS index."""
        index_file = output_dir / "embeddings.faiss"
        faiss.write_index(self._faiss_index, str(index_file))
        return index_file.stat().st_size
    
    def _save_numpy(self, output_dir: Path) -> int:
        """Save as NumPy binary."""
        index_file = output_dir / "embeddings.npy"
        np.save(str(index_file), self.embeddings)
        return index_file.stat().st_size
    
    def _save_json(self, output_dir: Path) -> int:
        """Save as JSON (fallback)."""
        index_file = output_dir / "embeddings.json"
        
        data = {
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else [],
            "chunk_ids": self.chunk_ids,
            "dimensions": self.dimensions
        }
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        
        return index_file.stat().st_size
    
    @classmethod
    def load(cls, index_dir: Path) -> 'VectorIndex':
        """
        Load an index from disk.
        
        Args:
            index_dir: Directory containing index files
            
        Returns:
            Loaded VectorIndex
        """
        metadata_file = index_dir / "index_metadata.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        format_str = metadata.get("format", "json")
        dimensions = metadata.get("dimensions", 0)
        chunk_ids = metadata.get("chunk_ids", [])
        
        # Load embeddings based on format
        embeddings = None
        
        if format_str == "faiss":
            faiss_file = index_dir / "embeddings.faiss"
            if faiss_file.exists() and FAISS_AVAILABLE and faiss is not None:
                faiss_index = faiss.read_index(str(faiss_file))
                # Batch reconstruct all vectors at once (O(1) calls instead of O(n))
                embeddings = faiss_index.reconstruct_n(0, faiss_index.ntotal)
        
        elif format_str == "numpy":
            numpy_file = index_dir / "embeddings.npy"
            if numpy_file.exists():
                embeddings = np.load(str(numpy_file))
        
        else:  # json
            json_file = index_dir / "embeddings.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                embeddings = np.array(data.get("embeddings", []), dtype=np.float32)
        
        # Load chunks
        chunks = {}
        chunks_file = index_dir / "chunks.jsonl"
        if chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        chunk_data = json.loads(line)
                        from .chunker import Chunk, ContentType
                        chunk = Chunk(
                            id=chunk_data["id"],
                            content=chunk_data["content"],
                            source_file=chunk_data["source_file"],
                            content_type=ContentType(chunk_data["content_type"]),
                            chunk_index=chunk_data["chunk_index"],
                            total_chunks=chunk_data["total_chunks"],
                            metadata=chunk_data.get("metadata", {}),
                            start_line=chunk_data.get("start_line"),
                            end_line=chunk_data.get("end_line"),
                            section=chunk_data.get("section")
                        )
                        chunks[chunk.id] = chunk
        
        return cls(
            embeddings=embeddings,
            chunk_ids=chunk_ids,
            chunks=chunks,
            dimensions=dimensions
        )


class IndexBuilder:
    """
    Builds vector indexes from embedding batches.
    """
    
    def __init__(self, use_faiss: bool = True):
        self.use_faiss = use_faiss
    
    def build(
        self,
        embedding_batch: EmbeddingBatch,
        chunks: List[Chunk]
    ) -> VectorIndex:
        """
        Build a vector index from embeddings.
        
        Args:
            embedding_batch: Batch of embeddings
            chunks: Original chunks (for metadata)
            
        Returns:
            VectorIndex ready for searching
        """
        # Create chunk lookup
        chunk_lookup = {chunk.id: chunk for chunk in chunks}
        
        # Get embeddings in order
        embeddings = embedding_batch.to_numpy()
        chunk_ids = embedding_batch.get_chunk_ids()
        
        return VectorIndex(
            embeddings=embeddings,
            chunk_ids=chunk_ids,
            chunks=chunk_lookup,
            dimensions=embedding_batch.dimensions,
            use_faiss=self.use_faiss
        )


def build_index(
    embedding_batch: EmbeddingBatch,
    chunks: List[Chunk],
    use_faiss: bool = True
) -> VectorIndex:
    """
    Convenience function to build an index.
    
    Args:
        embedding_batch: Embeddings
        chunks: Source chunks
        use_faiss: Whether to use FAISS
        
    Returns:
        VectorIndex
    """
    builder = IndexBuilder(use_faiss=use_faiss)
    return builder.build(embedding_batch, chunks)

