"""
Chunking utilities for splitting large text chunks while preserving relationships.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import re


@dataclass
class TextChunk:
    """A text chunk with metadata for RAG."""
    id: str
    text: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    chunk_index: Optional[int] = None  # For chunks split from same parent


class Chunker:
    """Chunks text while preserving hierarchical relationships."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            separators: List of separators to prefer when splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ['\n\n', '\n', '. ', ' ', '']
    
    def split_text(self, text: str, chunk_id: str) -> List[TextChunk]:
        """
        Split text into smaller chunks.
        
        Args:
            text: Text to split
            chunk_id: Base ID for the chunks
            
        Returns:
            List of TextChunk objects
        """
        if len(text) <= self.chunk_size:
            return [TextChunk(
                id=chunk_id,
                text=text,
                chunk_index=0
            )]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Determine end position
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunk_text = text[start:]
            else:
                # Try to split at a separator
                chunk_text = text[start:end]
                
                # Look for the best separator to split on
                best_split = -1
                for separator in self.separators:
                    if separator:
                        # Find last occurrence of separator
                        split_pos = chunk_text.rfind(separator)
                        if split_pos > self.chunk_size * 0.5:  # Don't split too early
                            best_split = split_pos + len(separator)
                            break
                
                if best_split > 0:
                    chunk_text = text[start:start + best_split]
                    end = start + best_split
                else:
                    # Force split at word boundary if possible
                    words = chunk_text.split()
                    if len(words) > 1:
                        # Remove last word and add it back in next chunk
                        chunk_text = ' '.join(words[:-1])
                        end = start + len(chunk_text)
            
            # Create chunk
            chunk = TextChunk(
                id=f"{chunk_id}_part_{chunk_index}",
                text=chunk_text.strip(),
                chunk_index=chunk_index
            )
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            chunk_index += 1
        
        # Link chunks together
        for i in range(len(chunks) - 1):
            chunks[i].children_ids = [chunks[i + 1].id]
            chunks[i + 1].parent_id = chunks[i].id
        
        return chunks
    
    def chunk_with_relationships(
        self,
        chunks: List[Dict],
        max_chunk_size: int = 1000
    ) -> List[TextChunk]:
        """
        Process chunks and split large ones while preserving relationships.
        
        Args:
            chunks: List of chunk dictionaries from parser
            max_chunk_size: Maximum size for a chunk before splitting
            
        Returns:
            List of TextChunk objects with relationships preserved
        """
        result_chunks = []
        chunk_map = {}  # Map original chunk IDs to new chunk IDs
        
        for chunk_dict in chunks:
            original_id = chunk_dict['id']
            text = chunk_dict['text']
            parent_id = chunk_dict.get('parent_id')
            children_ids = chunk_dict.get('children_ids', [])
            metadata = chunk_dict.get('metadata', {})
            
            # Split if too large
            if len(text) > max_chunk_size:
                split_chunks = self.split_text(text, original_id)
                
                # First chunk gets the parent relationship
                if split_chunks:
                    split_chunks[0].parent_id = parent_id
                    split_chunks[0].metadata = metadata.copy()
                    
                    # Last chunk gets the children relationships
                    if len(split_chunks) > 1:
                        split_chunks[-1].children_ids = children_ids.copy()
                    
                    # Add all split chunks
                    for split_chunk in split_chunks:
                        result_chunks.append(split_chunk)
                        chunk_map[original_id] = split_chunks[0].id  # Map to first chunk
            else:
                # Keep original chunk
                text_chunk = TextChunk(
                    id=original_id,
                    text=text,
                    parent_id=parent_id,
                    children_ids=children_ids.copy(),
                    metadata=metadata.copy()
                )
                result_chunks.append(text_chunk)
                chunk_map[original_id] = original_id
        
        # Update parent/child references to use new IDs
        for chunk in result_chunks:
            if chunk.parent_id and chunk.parent_id in chunk_map:
                chunk.parent_id = chunk_map[chunk.parent_id]
            
            # Update children IDs
            updated_children = []
            for child_id in chunk.children_ids:
                if child_id in chunk_map:
                    updated_children.append(chunk_map[child_id])
            chunk.children_ids = updated_children
        
        return result_chunks


def chunk_for_rag(chunks: List[Dict], chunk_size: int = 1000) -> List[Dict]:
    """
    Prepare chunks for RAG ingestion.
    
    Args:
        chunks: List of chunk dictionaries
        chunk_size: Target chunk size
        
    Returns:
        List of dictionaries ready for RAG indexing
    """
    chunker = Chunker(chunk_size=chunk_size)
    text_chunks = chunker.chunk_with_relationships(chunks, max_chunk_size=chunk_size)
    
    # Convert to dictionaries
    rag_chunks = []
    for chunk in text_chunks:
        rag_chunk = {
            'id': chunk.id,
            'text': chunk.text,
            'metadata': {
                **chunk.metadata,
                'parent_id': chunk.parent_id,
                'children_ids': chunk.children_ids,
                'chunk_index': chunk.chunk_index
            }
        }
        rag_chunks.append(rag_chunk)
    
    return rag_chunks

