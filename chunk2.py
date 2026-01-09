"""
Alternative chunking utilities that keep contiguous text together with overlapping chunks.
Uses a sliding window approach to create overlapping chunks while preserving natural text boundaries.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import re


@dataclass
class ContiguousChunk:
    """A contiguous text chunk with metadata for RAG."""
    id: str
    text: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    chunk_index: Optional[int] = None  # For chunks split from same parent
    start_char: Optional[int] = None  # Character position in original text
    end_char: Optional[int] = None  # Character position in original text


class ContiguousChunker:
    """Chunks text while keeping contiguous parts together with overlap."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            min_chunk_size: Minimum size for a chunk (smaller chunks will be merged)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def _find_sentence_boundaries(self, text: str) -> List[int]:
        """
        Find sentence boundaries in text.
        Returns list of character positions where sentences end.
        """
        # Pattern to match sentence endings (period, exclamation, question mark)
        # Followed by space or end of string
        pattern = r'[.!?]+(?:\s+|$)'
        boundaries = []
        
        for match in re.finditer(pattern, text):
            # Position after the sentence ending
            pos = match.end()
            boundaries.append(pos)
        
        # Always include the end of the text
        if boundaries and boundaries[-1] != len(text):
            boundaries.append(len(text))
        
        return boundaries
    
    def _find_paragraph_boundaries(self, text: str) -> List[int]:
        """
        Find paragraph boundaries (double newlines).
        Returns list of character positions where paragraphs end.
        """
        boundaries = []
        pattern = r'\n\n+'
        
        for match in re.finditer(pattern, text):
            pos = match.end()
            boundaries.append(pos)
        
        # Always include the end of the text
        if boundaries and boundaries[-1] != len(text):
            boundaries.append(len(text))
        
        return boundaries
    
    def _find_natural_boundaries(self, text: str) -> List[int]:
        """
        Find natural text boundaries (paragraphs, then sentences).
        Returns sorted list of character positions.
        """
        # First try paragraph boundaries
        para_boundaries = self._find_paragraph_boundaries(text)
        
        # If we have good paragraph boundaries, use those
        if len(para_boundaries) > 1:
            return para_boundaries
        
        # Otherwise use sentence boundaries
        sent_boundaries = self._find_sentence_boundaries(text)
        
        # If we have sentence boundaries, use those
        if len(sent_boundaries) > 1:
            return sent_boundaries
        
        # Fallback: use the end of text
        return [len(text)]
    
    def split_text_contiguous(
        self, 
        text: str, 
        chunk_id: str,
        metadata: Dict = None
    ) -> List[ContiguousChunk]:
        """
        Split text into contiguous chunks with overlap.
        Keeps text together at natural boundaries (paragraphs/sentences).
        
        Args:
            text: Text to split
            chunk_id: Base ID for the chunks
            metadata: Metadata to attach to chunks
            
        Returns:
            List of ContiguousChunk objects
        """
        if not text or len(text.strip()) == 0:
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # If text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            chunk = ContiguousChunk(
                id=chunk_id,
                text=text,
                metadata=metadata,
                chunk_index=0,
                start_char=0,
                end_char=len(text)
            )
            return [chunk]
        
        # Find natural boundaries
        boundaries = self._find_natural_boundaries(text)
        
        # Calculate step size (chunk_size - overlap)
        step_size = self.chunk_size - self.chunk_overlap
        
        # Create overlapping chunks using sliding window
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # Try to align end to a natural boundary
            # Find the closest boundary before or at the end position
            best_end = end
            for boundary in boundaries:
                if boundary <= end and boundary > start:
                    best_end = boundary
                elif boundary > end:
                    break
            
            # Extract chunk text
            chunk_text = text[start:best_end].strip()
            
            # Only create chunk if it meets minimum size
            if len(chunk_text) >= self.min_chunk_size or start == 0:
                chunk = ContiguousChunk(
                    id=f"{chunk_id}_part_{chunk_index}",
                    text=chunk_text,
                    metadata=metadata.copy(),
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=best_end
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start forward by step_size, but ensure we have overlap
            # If we're at the end, break
            if best_end >= len(text):
                break
            
            # Move start forward, ensuring overlap
            next_start = start + step_size
            
            # Try to align next_start to a natural boundary
            best_next_start = next_start
            for boundary in boundaries:
                if boundary <= next_start and boundary > start:
                    best_next_start = boundary
                elif boundary > next_start:
                    # Use the previous boundary if it's not too far back
                    if boundary - step_size <= next_start:
                        best_next_start = max(start + self.min_chunk_size, boundary - self.chunk_overlap)
                    break
            
            start = best_next_start
            
            # Safety check: ensure we're making progress
            if start <= chunks[-1].start_char if chunks else 0:
                start = chunks[-1].end_char - self.chunk_overlap if chunks else step_size
        
        return chunks
    
    def chunk_with_relationships(
        self,
        chunks: List[Dict],
        max_chunk_size: int = 1000
    ) -> List[ContiguousChunk]:
        """
        Process chunks and split large ones while preserving relationships.
        
        Args:
            chunks: List of chunk dictionaries from XML parser
            max_chunk_size: Maximum size before splitting
            
        Returns:
            List of ContiguousChunk objects
        """
        result_chunks = []
        chunk_id_map = {}  # Map original IDs to new chunk IDs
        
        for chunk_dict in chunks:
            original_id = chunk_dict.get('id')
            text = chunk_dict.get('text', '')
            parent_id = chunk_dict.get('parent_id')
            children_ids = chunk_dict.get('children_ids', [])
            metadata = chunk_dict.get('metadata', {})
            
            # Add element type and identifier to metadata
            metadata['tag'] = chunk_dict.get('element_type', '')
            metadata['identifier'] = chunk_dict.get('identifier')
            metadata['heading'] = metadata.get('heading', '')
            metadata['element_id'] = chunk_dict.get('id')
            
            # If text is small enough, keep as single chunk
            if len(text) <= max_chunk_size:
                new_chunk = ContiguousChunk(
                    id=original_id,
                    text=text,
                    parent_id=parent_id,
                    children_ids=children_ids.copy(),
                    metadata=metadata,
                    chunk_index=None,
                    start_char=0,
                    end_char=len(text)
                )
                result_chunks.append(new_chunk)
                chunk_id_map[original_id] = [original_id]
            else:
                # Split into multiple chunks
                split_chunks = self.split_text_contiguous(
                    text, 
                    original_id,
                    metadata
                )
                
                if split_chunks:
                    # First chunk keeps the original ID and relationships
                    split_chunks[0].id = original_id
                    split_chunks[0].parent_id = parent_id
                    split_chunks[0].children_ids = children_ids.copy()
                    split_chunks[0].chunk_index = None
                    
                    # Subsequent chunks get new IDs
                    new_ids = [original_id]
                    for i, split_chunk in enumerate(split_chunks[1:], 1):
                        split_chunk.id = f"{original_id}_part_{i}"
                        split_chunk.parent_id = original_id  # Parent is the first chunk
                        split_chunk.chunk_index = i
                        new_ids.append(split_chunk.id)
                    
                    result_chunks.extend(split_chunks)
                    chunk_id_map[original_id] = new_ids
                else:
                    # Fallback: create single chunk even if large
                    new_chunk = ContiguousChunk(
                        id=original_id,
                        text=text,
                        parent_id=parent_id,
                        children_ids=children_ids.copy(),
                        metadata=metadata,
                        chunk_index=None,
                        start_char=0,
                        end_char=len(text)
                    )
                    result_chunks.append(new_chunk)
                    chunk_id_map[original_id] = [original_id]
        
        # Update parent/child relationships
        for chunk in result_chunks:
            if chunk.parent_id and chunk.parent_id in chunk_id_map:
                # Update parent_id to first chunk of parent
                chunk.parent_id = chunk_id_map[chunk.parent_id][0]
            
            # Update children_ids
            updated_children = []
            for child_id in chunk.children_ids:
                if child_id in chunk_id_map:
                    # Add all chunks from the child
                    updated_children.extend(chunk_id_map[child_id])
                else:
                    updated_children.append(child_id)
            chunk.children_ids = updated_children
        
        return result_chunks


def chunk_for_rag_contiguous(
    chunks: List[Dict], 
    chunk_size: int = 2000, 
    chunk_overlap: int = 400
) -> List[Dict]:
    """
    Prepare chunks for RAG ingestion using contiguous chunking with overlap.
    
    Args:
        chunks: List of chunk dictionaries
        chunk_size: Target chunk size (in characters)
        chunk_overlap: Overlap between chunks (in characters)
        
    Returns:
        List of dictionaries ready for RAG indexing
    """
    chunker = ContiguousChunker(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
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
                'chunk_index': chunk.chunk_index,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char,
            }
        }
        rag_chunks.append(rag_chunk)
    
    return rag_chunks

