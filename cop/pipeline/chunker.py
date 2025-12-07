"""
Content Chunker for COP Packages

Smart chunking for different content types (Markdown, YAML, JSON)
with semantic boundary awareness.
"""

import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator
from enum import Enum

from .resolver import ResolvedFile, ContentType


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    SEMANTIC = "semantic"    # Split on semantic boundaries (headers, sections)
    FIXED = "fixed"          # Fixed token/character count
    PARAGRAPH = "paragraph"  # Split on paragraphs
    HYBRID = "hybrid"        # Combine semantic + size limits


@dataclass
class Chunk:
    """A content chunk ready for embedding."""
    id: str                          # Unique identifier
    content: str                     # The actual text content
    source_file: str                 # Original file path
    content_type: ContentType        # Type of source content
    chunk_index: int                 # Index within the source file
    total_chunks: int                # Total chunks from this file
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Positioning info
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    section: Optional[str] = None    # Section header if applicable
    
    @property
    def char_count(self) -> int:
        return len(self.content)
    
    @property 
    def estimated_tokens(self) -> int:
        """Rough token estimate (avg 4 chars per token)."""
        return self.char_count // 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "source_file": self.source_file,
            "content_type": self.content_type.value,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "char_count": self.char_count,
            "estimated_tokens": self.estimated_tokens,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "section": self.section,
            "metadata": self.metadata
        }


@dataclass 
class ChunkerConfig:
    """Configuration for the content chunker."""
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    max_chunk_size: int = 512         # Max tokens per chunk
    min_chunk_size: int = 50          # Min tokens per chunk
    overlap: int = 50                 # Token overlap between chunks
    preserve_code_blocks: bool = True # Keep code blocks intact
    include_headers: bool = True      # Include section headers in chunks


class ContentChunker:
    """
    Chunks content from resolved files into embedding-ready pieces.
    """
    
    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.config = config or ChunkerConfig()
        self._chunk_counter = 0
    
    def chunk_file(self, resolved_file: ResolvedFile) -> List[Chunk]:
        """
        Chunk a single resolved file.
        
        Args:
            resolved_file: The file to chunk
            
        Returns:
            List of Chunk objects
        """
        if resolved_file.is_markdown:
            return self._chunk_markdown(resolved_file)
        elif resolved_file.is_yaml:
            return self._chunk_yaml(resolved_file)
        elif resolved_file.is_json:
            return self._chunk_json(resolved_file)
        else:
            return self._chunk_plain_text(resolved_file)
    
    def chunk_files(self, files: List[ResolvedFile]) -> List[Chunk]:
        """
        Chunk multiple files.
        
        Args:
            files: List of resolved files
            
        Returns:
            All chunks from all files
        """
        all_chunks = []
        for f in files:
            all_chunks.extend(self.chunk_file(f))
        return all_chunks
    
    def _chunk_markdown(self, resolved_file: ResolvedFile) -> List[Chunk]:
        """Chunk markdown content with semantic awareness."""
        content = resolved_file.content
        sections = self._split_markdown_sections(content)
        
        chunks = []
        for section in sections:
            section_chunks = self._create_chunks_from_section(
                section["content"],
                resolved_file,
                section_header=section.get("header"),
                start_line=section.get("start_line")
            )
            chunks.extend(section_chunks)
        
        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _split_markdown_sections(self, content: str) -> List[Dict]:
        """Split markdown into sections based on headers."""
        sections = []
        lines = content.split('\n')
        
        current_section = {
            "header": None,
            "content": "",
            "start_line": 1
        }
        
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        
        for i, line in enumerate(lines, 1):
            match = header_pattern.match(line)
            if match:
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "header": match.group(2).strip(),
                    "header_level": len(match.group(1)),
                    "content": line + "\n",
                    "start_line": i
                }
            else:
                current_section["content"] += line + "\n"
        
        # Don't forget the last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        # If no sections found, return entire content as one section
        if not sections:
            sections = [{"header": None, "content": content, "start_line": 1}]
        
        return sections
    
    def _chunk_yaml(self, resolved_file: ResolvedFile) -> List[Chunk]:
        """Chunk YAML content by top-level keys."""
        chunks = []
        parsed = resolved_file.parsed
        
        if not isinstance(parsed, dict):
            # Not a dict, chunk as plain text
            return self._chunk_plain_text(resolved_file)
        
        # Create chunks for meaningful sections
        for key, value in parsed.items():
            section_content = self._yaml_section_to_text(key, value)
            
            if len(section_content) > self.config.min_chunk_size * 4:  # chars
                chunk = self._create_chunk(
                    content=section_content,
                    resolved_file=resolved_file,
                    section=key,
                    chunk_index=len(chunks)
                )
                chunks.append(chunk)
        
        # If no meaningful chunks, create one for the whole file
        if not chunks:
            chunk = self._create_chunk(
                content=resolved_file.content,
                resolved_file=resolved_file,
                chunk_index=0
            )
            chunks.append(chunk)
        
        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _yaml_section_to_text(self, key: str, value: Any) -> str:
        """Convert a YAML section to readable text."""
        lines = [f"## {key}"]
        
        if isinstance(value, str):
            lines.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for k, v in item.items():
                        lines.append(f"- {k}: {v}")
                else:
                    lines.append(f"- {item}")
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (str, int, float, bool)):
                    lines.append(f"- {k}: {v}")
                elif isinstance(v, list):
                    lines.append(f"- {k}:")
                    for item in v[:5]:  # Limit list items
                        lines.append(f"  - {item}")
                elif isinstance(v, dict):
                    lines.append(f"- {k}:")
                    for sub_k, sub_v in list(v.items())[:5]:
                        lines.append(f"  - {sub_k}: {sub_v}")
        
        return "\n".join(lines)
    
    def _chunk_json(self, resolved_file: ResolvedFile) -> List[Chunk]:
        """Chunk JSON content."""
        parsed = resolved_file.parsed
        
        chunks = []
        
        if isinstance(parsed, list):
            # Array of items - chunk by items
            for i, item in enumerate(parsed):
                item_text = self._json_item_to_text(item, index=i)
                if len(item_text) >= self.config.min_chunk_size * 4:
                    chunk = self._create_chunk(
                        content=item_text,
                        resolved_file=resolved_file,
                        section=f"item_{i}",
                        chunk_index=len(chunks)
                    )
                    chunks.append(chunk)
        elif isinstance(parsed, dict):
            # Object - chunk by top-level keys
            for key, value in parsed.items():
                section_text = self._json_item_to_text({key: value})
                if len(section_text) >= self.config.min_chunk_size * 4:
                    chunk = self._create_chunk(
                        content=section_text,
                        resolved_file=resolved_file,
                        section=key,
                        chunk_index=len(chunks)
                    )
                    chunks.append(chunk)
        
        if not chunks:
            # Fallback: entire file as one chunk
            chunk = self._create_chunk(
                content=json.dumps(parsed, indent=2) if parsed else resolved_file.content,
                resolved_file=resolved_file,
                chunk_index=0
            )
            chunks.append(chunk)
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _json_item_to_text(self, item: Any, index: Optional[int] = None) -> str:
        """Convert a JSON item to readable text."""
        lines = []
        
        if index is not None:
            lines.append(f"Item {index + 1}:")
        
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    lines.append(f"  {key}: {value}")
                elif isinstance(value, list):
                    lines.append(f"  {key}: [{', '.join(str(v) for v in value[:5])}]")
                elif isinstance(value, dict):
                    lines.append(f"  {key}: {json.dumps(value)[:100]}...")
        else:
            lines.append(str(item))
        
        return "\n".join(lines)
    
    def _chunk_plain_text(self, resolved_file: ResolvedFile) -> List[Chunk]:
        """Chunk plain text by paragraphs or fixed size."""
        content = resolved_file.content
        
        if self.config.strategy == ChunkingStrategy.PARAGRAPH:
            paragraphs = re.split(r'\n\s*\n', content)
            chunks = []
            
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if para and len(para) >= self.config.min_chunk_size * 4:
                    chunk = self._create_chunk(
                        content=para,
                        resolved_file=resolved_file,
                        chunk_index=i
                    )
                    chunks.append(chunk)
            
            if not chunks:
                chunk = self._create_chunk(
                    content=content,
                    resolved_file=resolved_file,
                    chunk_index=0
                )
                chunks.append(chunk)
        else:
            # Fixed size chunking
            chunks = self._fixed_size_chunk(content, resolved_file)
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _fixed_size_chunk(
        self, 
        content: str, 
        resolved_file: ResolvedFile
    ) -> List[Chunk]:
        """Create fixed-size chunks with overlap."""
        max_chars = self.config.max_chunk_size * 4  # Rough char estimate
        overlap_chars = self.config.overlap * 4
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + max_chars
            
            # Try to break at sentence or word boundary
            if end < len(content):
                # Look for sentence end
                sentence_end = content.rfind('. ', start, end)
                if sentence_end > start + max_chars // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = content.rfind(' ', start, end)
                    if word_end > start:
                        end = word_end
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                chunk = self._create_chunk(
                    content=chunk_content,
                    resolved_file=resolved_file,
                    chunk_index=len(chunks)
                )
                chunks.append(chunk)
            
            start = end - overlap_chars if end < len(content) else len(content)
        
        return chunks
    
    def _create_chunks_from_section(
        self,
        content: str,
        resolved_file: ResolvedFile,
        section_header: Optional[str] = None,
        start_line: Optional[int] = None
    ) -> List[Chunk]:
        """Create chunks from a section, respecting size limits."""
        max_chars = self.config.max_chunk_size * 4
        
        # If section fits in one chunk
        if len(content) <= max_chars:
            chunk = self._create_chunk(
                content=content.strip(),
                resolved_file=resolved_file,
                section=section_header,
                start_line=start_line,
                chunk_index=0
            )
            return [chunk]
        
        # Split into multiple chunks
        chunks = []
        
        # Try to split on paragraphs first
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = ""
        if section_header and self.config.include_headers:
            current_chunk = f"# {section_header}\n\n"
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) + 2 <= max_chars:
                current_chunk += para + "\n\n"
            else:
                # Save current chunk
                if current_chunk.strip():
                    chunk = self._create_chunk(
                        content=current_chunk.strip(),
                        resolved_file=resolved_file,
                        section=section_header,
                        chunk_index=len(chunks)
                    )
                    chunks.append(chunk)
                
                # Start new chunk
                current_chunk = ""
                if section_header and self.config.include_headers:
                    current_chunk = f"# {section_header} (continued)\n\n"
                current_chunk += para + "\n\n"
        
        # Don't forget last chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                content=current_chunk.strip(),
                resolved_file=resolved_file,
                section=section_header,
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        resolved_file: ResolvedFile,
        section: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        chunk_index: int = 0
    ) -> Chunk:
        """Create a chunk with a unique ID."""
        self._chunk_counter += 1
        
        chunk_id = f"{resolved_file.relative_path}#{chunk_index}"
        
        return Chunk(
            id=chunk_id,
            content=content,
            source_file=resolved_file.relative_path,
            content_type=resolved_file.content_type,
            chunk_index=chunk_index,
            total_chunks=1,  # Updated later
            metadata={
                **resolved_file.metadata,
                "file_type": resolved_file.extension
            },
            start_line=start_line,
            end_line=end_line,
            section=section
        )


def chunk_resolved_files(
    files: List[ResolvedFile],
    config: Optional[ChunkerConfig] = None
) -> List[Chunk]:
    """
    Convenience function to chunk multiple files.
    
    Args:
        files: List of resolved files
        config: Optional chunking configuration
        
    Returns:
        List of all chunks
    """
    chunker = ContentChunker(config)
    return chunker.chunk_files(files)

