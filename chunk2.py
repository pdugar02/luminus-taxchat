"""
Structure-first chunking utilities for tax code / legal authority RAG.
Chunks by legal boundaries (paragraph/subparagraph) first, then adjusts by token count.
Uses tiktoken for accurate token counting.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import re
import tiktoken


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


class StructureFirstChunker:
    """
    Chunks text by legal structure first, then adjusts by token count.
    Target: 300-700 tokens per chunk (sweet spot)
    Overlap: 0-80 tokens (often none if chunking by legal boundaries)
    """
    
    def __init__(
        self,
        target_tokens: int = 500,  # Sweet spot: 300-700, default to middle
        min_tokens: int = 300,
        max_tokens: int = 700,
        split_threshold: int = 1000,  # Split if >1000 tokens
        merge_threshold: int = 150,  # Merge if <150 tokens
        chunk_overlap: int = 50,  # 0-80 tokens overlap
        model: str = "gpt-4"  # Model for tiktoken encoding
    ):
        """
        Initialize chunker.
        
        Args:
            target_tokens: Target token count for chunks (default: 500, sweet spot 300-700)
            min_tokens: Minimum tokens before merging (default: 300)
            max_tokens: Maximum tokens before splitting (default: 700)
            split_threshold: Split chunks larger than this (default: 1000)
            merge_threshold: Merge chunks smaller than this (default: 150)
            chunk_overlap: Overlap between chunks in tokens (default: 50, range 0-80)
            model: Model name for tiktoken encoding (default: "gpt-4")
        """
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        self.chunk_overlap = chunk_overlap
        
        # Initialize tiktoken encoder
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base (used by GPT-4)
            self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoder.encode(text))
    
    def _find_legal_subitem_boundaries(self, text: str) -> List[Tuple[int, str]]:
        """
        Find legal sub-item boundaries like (A), (B), (C) or (i), (ii), (iii).
        Returns list of (position, label) tuples.
        """
        boundaries = []
        
        # Pattern for sub-items: (A), (B), (C), etc. or (i), (ii), (iii), etc.
        # Also matches (1), (2), (3) as sub-items
        patterns = [
            r'\(([A-Z])\)',  # (A), (B), (C)
            r'\(([ivxlcdm]+)\)',  # (i), (ii), (iii) - roman numerals
            r'\(([IVXLCDM]+)\)',  # (I), (II), (III) - uppercase roman
            r'\((\d+)\)',  # (1), (2), (3) - numbers
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                # Position at the start of the sub-item marker
                pos = match.start()
                label = match.group(0)
                boundaries.append((pos, label))
        
        # Sort by position
        boundaries.sort(key=lambda x: x[0])
        
        # Remove duplicates (same position)
        unique_boundaries = []
        seen_positions = set()
        for pos, label in boundaries:
            if pos not in seen_positions:
                unique_boundaries.append((pos, label))
                seen_positions.add(pos)
        
        return unique_boundaries
    
    def _find_subparagraph_boundaries(self, text: str) -> List[int]:
        """
        Find subparagraph boundaries like (a)(1), (a)(2), etc.
        Returns list of character positions.
        """
        boundaries = []
        
        # Pattern for subparagraphs: (a)(1), (a)(2), (b)(1), etc.
        # Look for patterns like (letter)(number) that appear at start of line or after space
        pattern = r'(?:^|\s)\(([a-z])\)\((\d+)\)'
        
        for match in re.finditer(pattern, text, re.MULTILINE):
            # Position at the start of the subparagraph marker
            pos = match.start()
            boundaries.append(pos)
        
        # Also look for standalone paragraph markers like (a), (b), (c)
        pattern2 = r'(?:^|\s)\(([a-z])\)(?!\()'
        for match in re.finditer(pattern2, text, re.MULTILINE):
            pos = match.start()
            if pos not in boundaries:
                boundaries.append(pos)
        
        boundaries.sort()
        return boundaries
    
    def _split_by_subitems(self, text: str, chunk_id: str, metadata: Dict) -> List[ContiguousChunk]:
        """
        Split a large chunk (>split_threshold tokens) by sub-items.
        Returns list of chunks split at sub-item boundaries.
        """
        token_count = self.count_tokens(text)
        
        # If not large enough to split, return as single chunk
        if token_count <= self.split_threshold:
            return [ContiguousChunk(
                id=chunk_id,
                text=text,
                metadata=metadata,
                chunk_index=None,
                start_char=0,
                end_char=len(text)
            )]
        
        # Find sub-item boundaries
        subitem_boundaries = self._find_legal_subitem_boundaries(text)
        
        # If no sub-item boundaries found, try subparagraph boundaries
        if not subitem_boundaries:
            subpara_boundaries = self._find_subparagraph_boundaries(text)
            if subpara_boundaries:
                # Convert to (pos, label) format
                subitem_boundaries = [(pos, f"subpara_{i}") for i, pos in enumerate(subpara_boundaries)]
        
        # If still no boundaries, try sentence boundaries as last resort
        if not subitem_boundaries:
            # Find sentence boundaries
            pattern = r'[.!?]+(?:\s+|$)'
            for match in re.finditer(pattern, text):
                pos = match.end()
                subitem_boundaries.append((pos, f"sentence_{len(subitem_boundaries)}"))
        
        if not subitem_boundaries:
            # Can't split, return as single chunk
            return [ContiguousChunk(
                id=chunk_id,
                text=text,
                metadata=metadata,
                chunk_index=None,
                start_char=0,
                end_char=len(text)
            )]
        
        # Split at boundaries, ensuring each chunk is within reasonable size
        chunks = []
        start = 0
        chunk_index = 0
        
        for i, (boundary_pos, label) in enumerate(subitem_boundaries):
            # Extract chunk from start to boundary
            chunk_text = text[start:boundary_pos].strip()
            
            if chunk_text:
                chunk_tokens = self.count_tokens(chunk_text)
                
                # If chunk is still too large, try to split further
                if chunk_tokens > self.split_threshold and i < len(subitem_boundaries) - 1:
                    # Try to find more granular boundaries within this chunk
                    # For now, just create the chunk and let it be large
                    pass
                
                chunk_id_use = chunk_id if chunk_index == 0 else f"{chunk_id}_part_{chunk_index}"
                chunk = ContiguousChunk(
                    id=chunk_id_use,
                    text=chunk_text,
                    metadata=metadata.copy(),
                    chunk_index=chunk_index if chunk_index > 0 else None,
                    start_char=start,
                    end_char=boundary_pos
                )
                chunks.append(chunk)
                chunk_index += 1
            
            start = boundary_pos
        
        # Add final chunk from last boundary to end
        if start < len(text):
            chunk_text = text[start:].strip()
            if chunk_text:
                chunk_id_use = chunk_id if chunk_index == 0 else f"{chunk_id}_part_{chunk_index}"
                chunk = ContiguousChunk(
                    id=chunk_id_use,
                    text=chunk_text,
                    metadata=metadata.copy(),
                    chunk_index=chunk_index if chunk_index > 0 else None,
                    start_char=start,
                    end_char=len(text)
                )
                chunks.append(chunk)
        
        return chunks if chunks else [ContiguousChunk(
            id=chunk_id,
            text=text,
            metadata=metadata,
            chunk_index=None,
            start_char=0,
            end_char=len(text)
        )]
    
    def _merge_small_chunks(
        self, 
        chunks: List[Dict], 
        start_idx: int
    ) -> Tuple[List[ContiguousChunk], int]:
        """
        Merge small chunks (<merge_threshold tokens) with next siblings until reaching target size.
        Only merges chunks that are siblings (same parent_id).
        Returns (merged_chunks, next_index).
        """
        if start_idx >= len(chunks):
            return [], start_idx
        
        first_chunk = chunks[start_idx]
        parent_id = first_chunk.get('parent_id')
        element_type = first_chunk.get('element_type', '')
        
        # Only merge paragraph/subparagraph level chunks
        if element_type not in ['paragraph', 'subparagraph']:
            return [], start_idx
        
        merged_text_parts = []
        merged_metadata = first_chunk.get('metadata', {}).copy()
        total_tokens = 0
        current_idx = start_idx
        chunk_ids = []
        children_ids = []
        
        # Start with first chunk
        while current_idx < len(chunks):
            chunk = chunks[current_idx]
            
            # Only merge if same parent (siblings)
            if chunk.get('parent_id') != parent_id:
                break
            
            # Only merge if same element type
            if chunk.get('element_type', '') != element_type:
                break
            
            chunk_text = chunk.get('text', '')
            chunk_tokens = self.count_tokens(chunk_text)
            
            # If adding this chunk would exceed max_tokens, stop
            if total_tokens + chunk_tokens > self.max_tokens and total_tokens >= self.min_tokens:
                break
            
            # Add this chunk
            merged_text_parts.append(chunk_text)
            total_tokens += chunk_tokens
            chunk_ids.append(chunk.get('id'))
            children_ids.extend(chunk.get('children_ids', []))
            current_idx += 1
            
            # If we've reached target size, stop
            if total_tokens >= self.min_tokens:
                break
        
        # Only merge if we actually merged multiple chunks
        if current_idx <= start_idx + 1:
            return [], start_idx
        
        # Combine text with newlines
        merged_text = '\n\n'.join(merged_text_parts)
        
        # Use first chunk's ID as the merged chunk ID
        merged_id = chunk_ids[0] if chunk_ids else first_chunk.get('id')
        
        # Create merged chunk
        merged_chunk = ContiguousChunk(
            id=merged_id,
            text=merged_text,
            parent_id=parent_id,
            children_ids=children_ids,
            metadata=merged_metadata,
            chunk_index=None,
            start_char=0,
            end_char=len(merged_text)
        )
        
        return [merged_chunk], current_idx
    
    def chunk_with_relationships(
        self,
        chunks: List[Dict],
        max_chunk_size: Optional[int] = None  # Deprecated, kept for compatibility
    ) -> List[ContiguousChunk]:
        """
        Process chunks using structure-first chunking strategy.
        
        Strategy:
        1. Chunk by structure first (paragraph/subparagraph boundaries)
        2. If <merge_threshold tokens, merge with next sibling until ~min_tokens
        3. If >split_threshold tokens, split by sub-items
        4. Target: 300-700 tokens per chunk
        
        Args:
            chunks: List of chunk dictionaries from XML parser
            max_chunk_size: Deprecated, kept for compatibility
            
        Returns:
            List of ContiguousChunk objects
        """
        result_chunks = []
        chunk_id_map = {}  # Map original IDs to new chunk IDs
        
        i = 0
        while i < len(chunks):
            chunk_dict = chunks[i]
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
            
            token_count = self.count_tokens(text)
            element_type = chunk_dict.get('element_type', '')
            
            # Strategy: chunk by structure first, size second
            if token_count < self.merge_threshold:
                # Small chunk: try to merge with next siblings
                # Only merge if it's a paragraph/subparagraph level element
                if element_type in ['paragraph', 'subparagraph']:
                    merged_chunks, next_idx = self._merge_small_chunks(chunks, i)
                    if next_idx > i + 1:
                        # Successfully merged multiple chunks
                        result_chunks.extend(merged_chunks)
                        # Update chunk_id_map for all merged chunks
                        for orig_chunk in chunks[i:next_idx]:
                            orig_id = orig_chunk.get('id')
                            chunk_id_map[orig_id] = [merged_chunks[0].id]
                        i = next_idx
                        continue
                
                # Couldn't merge, keep as single chunk
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
                i += 1
                
            elif token_count > self.split_threshold:
                # Large chunk: split by sub-items
                split_chunks = self._split_by_subitems(text, original_id, metadata)
                
                if len(split_chunks) > 1:
                    # Successfully split
                    # First chunk keeps the original ID and relationships
                    split_chunks[0].parent_id = parent_id
                    split_chunks[0].children_ids = children_ids.copy()
                    split_chunks[0].chunk_index = None
                    
                    # Subsequent chunks get new IDs
                    new_ids = [original_id]
                    for j, split_chunk in enumerate(split_chunks[1:], 1):
                        split_chunk.parent_id = original_id  # Parent is the first chunk
                        split_chunk.chunk_index = j
                        new_ids.append(split_chunk.id)
                    
                    result_chunks.extend(split_chunks)
                    chunk_id_map[original_id] = new_ids
                else:
                    # Couldn't split, keep as single chunk
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
                i += 1
                
            else:
                # Chunk is in good size range (merge_threshold to split_threshold)
                # Keep as single chunk
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
                i += 1
        
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
    chunk_size: int = 500,  # Now in tokens, default 500 (sweet spot)
    chunk_overlap: int = 50  # Now in tokens, default 50 (range 0-80)
) -> List[Dict]:
    """
    Prepare chunks for RAG ingestion using structure-first chunking with token counting.
    
    Args:
        chunks: List of chunk dictionaries
        chunk_size: Target chunk size in tokens (default: 500, sweet spot 300-700)
        chunk_overlap: Overlap between chunks in tokens (default: 50, range 0-80)
        
    Returns:
        List of dictionaries ready for RAG indexing
    """
    # Calculate min/max from target
    min_tokens = max(300, int(chunk_size * 0.6))  # At least 300, or 60% of target
    max_tokens = min(700, int(chunk_size * 1.4))  # At most 700, or 140% of target
    
    chunker = StructureFirstChunker(
        target_tokens=chunk_size,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        split_threshold=1000,  # Split if >1000 tokens
        merge_threshold=150,  # Merge if <150 tokens
        chunk_overlap=chunk_overlap
    )
    
    text_chunks = chunker.chunk_with_relationships(chunks)
    
    # Convert to dictionaries
    rag_chunks = []
    for chunk in text_chunks:
        # Count tokens for metadata
        token_count = chunker.count_tokens(chunk.text)
        
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
                'token_count': token_count,  # Add token count to metadata
            }
        }
        rag_chunks.append(rag_chunk)
    
    return rag_chunks
