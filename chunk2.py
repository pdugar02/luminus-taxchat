"""
Structure-first chunking utilities for tax code / legal authority RAG.
Chunks by legal boundaries (paragraph/subparagraph) first, then adjusts by token count.
Uses tiktoken for accurate token counting.
Target: 300-700 tokens per chunk (sweet spot)
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
    chunk_index: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class StructureFirstChunker:
    """
    Chunks text by legal structure first, then adjusts by token count.
    Target: 300-700 tokens per chunk (sweet spot)
    Overlap: 0-80 tokens (often none if chunking by legal boundaries)
    """
    
    def __init__(
        self,
        target_tokens: int = 500,
        min_tokens: int = 300,
        max_tokens: int = 700,
        split_threshold: int = 700,  # Split if >700 tokens (not 1000!)
        merge_threshold: int = 150,
        chunk_overlap: int = 50,
        model: str = "gpt-4"
    ):
        """
        Initialize chunker.
        
        Args:
            target_tokens: Target token count (default: 500)
            min_tokens: Minimum tokens before merging (default: 300)
            max_tokens: Maximum tokens before splitting (default: 700)
            split_threshold: Split chunks larger than this (default: 700)
            merge_threshold: Merge chunks smaller than this (default: 150)
            chunk_overlap: Overlap between chunks in tokens (default: 50)
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
            self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoder.encode(text))
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing unnecessary whitespace and normalizing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize whitespace (multiple spaces/newlines to single)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove excessive punctuation (e.g., multiple periods)
        text = re.sub(r'\.{3,}', '...', text)
        
        return text
    
    def _find_subsection_boundaries(self, text: str) -> List[Tuple[int, str]]:
        """
        Find subsection boundaries like (a), (b), (c), etc. in US tax code sections.
        Pattern matches (lowercase_letter) followed by space and capital letter.
        Returns list of (position, subsection_label) tuples sorted by position.
        
        Args:
            text: Text to search for subsection boundaries
            
        Returns:
            List of (position, subsection_label) tuples, e.g., [(100, '(a)'), (500, '(b)')]
        """
        boundaries = []
        
        # Pattern: (lowercase_letter) followed by space and capital letter
        # This matches subsection markers like "(a) Married individuals..." or "(b) Heads of households..."
        pattern = r'\(([a-z])\)\s+([A-Z])'
        
        for match in re.finditer(pattern, text):
            pos = match.start()
            subsection_label = match.group(1)  # Extract just the letter, e.g., 'a', 'b'
            boundaries.append((pos, f"({subsection_label})"))
        
        # Sort by position
        boundaries.sort(key=lambda x: x[0])
        
        return boundaries
    
    def _find_legal_boundaries(self, text: str) -> List[Tuple[int, str]]:
        """
        Find legal sub-item boundaries like (A), (B), (C), (i), (ii), (a)(1), etc.
        Returns list of (position, label) tuples sorted by position.
        """
        boundaries = []
        
        # Patterns for legal boundaries (in order of specificity)
        patterns = [
            (r'\(([a-z])\)\((\d+)\)', 'subpara'),  # (a)(1), (a)(2) - most specific
            (r'\(([A-Z])\)', 'subitem'),  # (A), (B), (C)
            (r'\(([ivxlcdm]+)\)', 'roman_lower'),  # (i), (ii), (iii)
            (r'\(([IVXLCDM]+)\)', 'roman_upper'),  # (I), (II), (III)
            (r'\((\d+)\)', 'number'),  # (1), (2), (3)
            (r'\(([a-z])\)(?!\()', 'paragraph'),  # (a), (b) - standalone
        ]
        
        for pattern, label_type in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                pos = match.start()
                label = match.group(0)
                boundaries.append((pos, f"{label_type}:{label}"))
        
        # Sort by position and remove duplicates
        boundaries.sort(key=lambda x: x[0])
        unique_boundaries = []
        seen_positions = set()
        for pos, label in boundaries:
            if pos not in seen_positions:
                unique_boundaries.append((pos, label))
                seen_positions.add(pos)
        
        return unique_boundaries
    
    def _split_large_chunk(self, text: str, chunk_id: str, metadata: Dict, depth: int = 0) -> List[ContiguousChunk]:
        """
        Split a large chunk (>split_threshold tokens) by legal boundaries.
        Ensures resulting chunks are within max_tokens limit.
        
        Args:
            text: Text to split
            chunk_id: ID for the chunk
            metadata: Metadata dictionary
            depth: Recursion depth (prevents infinite recursion)
        """
        MAX_RECURSION_DEPTH = 5
        
        text = self._clean_text(text)
        token_count = self.count_tokens(text)
        
        # If within limits, return as single chunk
        if token_count <= self.split_threshold:
            return [ContiguousChunk(
                id=chunk_id,
                text=text,
                metadata=metadata,
                chunk_index=None,
                start_char=0,
                end_char=len(text)
            )]
        
        # Prevent infinite recursion - if we've recursed too deep, force split by characters
        if depth >= MAX_RECURSION_DEPTH:
            # Fallback: split by approximate character count
            # Estimate chars per token (roughly 4 chars per token for English)
            chars_per_token = 4
            target_chars = self.max_tokens * chars_per_token
            chunks = []
            chunk_index = 0
            start = 0
            
            while start < len(text):
                end = min(start + target_chars, len(text))
                # Try to break at sentence boundary
                if end < len(text):
                    # Look for sentence boundary near the end
                    for i in range(end, max(start, end - 200), -1):
                        if text[i] in '.!?':
                            end = i + 1
                            break
                
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunk = ContiguousChunk(
                        id=f"{chunk_id}_part_{chunk_index}",
                        text=chunk_text,
                        metadata=metadata.copy(),
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                start = end
            
            return chunks if chunks else [ContiguousChunk(
                id=chunk_id,
                text=text[:target_chars],  # Truncate if we can't split
                metadata=metadata,
                chunk_index=None,
                start_char=0,
                end_char=min(target_chars, len(text))
            )]
        
        # Find legal boundaries
        boundaries = self._find_legal_boundaries(text)
        
        # If no boundaries found, try sentence boundaries
        if not boundaries:
            for match in re.finditer(r'[.!?]+\s+', text):
                boundaries.append((match.end(), f"sentence:{match.end()}"))
        
        if not boundaries:
            # Can't split meaningfully - if too large, force character-based split
            if token_count > self.max_tokens:
                return self._split_large_chunk(text, chunk_id, metadata, depth + 1)
            # Otherwise return as single chunk (will be truncated later if needed)
            return [ContiguousChunk(
                id=chunk_id,
                text=text,
                metadata=metadata,
                chunk_index=None,
                start_char=0,
                end_char=len(text)
            )]
        
        # Split at boundaries, ensuring each chunk is within max_tokens
        chunks = []
        start = 0
        chunk_index = 0
        
        for i, (boundary_pos, label) in enumerate(boundaries):
            chunk_text = text[start:boundary_pos].strip()
            
            if not chunk_text:
                start = boundary_pos
                continue
            
            chunk_tokens = self.count_tokens(chunk_text)
            
            # If this chunk is still too large, split it further (with recursion limit)
            if chunk_tokens > self.max_tokens:
                # Recursively split this sub-chunk
                sub_chunks = self._split_large_chunk(
                    chunk_text,
                    f"{chunk_id}_part_{chunk_index}",
                    metadata.copy(),
                    depth + 1
                )
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_index = chunk_index
                    chunks.append(sub_chunk)
                    chunk_index += 1
            else:
                # Chunk is within limits
                chunk = ContiguousChunk(
                    id=chunk_id if chunk_index == 0 else f"{chunk_id}_part_{chunk_index}",
                    text=chunk_text,
                    metadata=metadata.copy(),
                    chunk_index=chunk_index if chunk_index > 0 else None,
                    start_char=start,
                    end_char=boundary_pos
                )
                chunks.append(chunk)
                chunk_index += 1
            
            start = boundary_pos
        
        # Add final chunk
        if start < len(text):
            final_text = text[start:].strip()
            if final_text:
                final_tokens = self.count_tokens(final_text)
                if final_tokens > self.max_tokens:
                    # Split final chunk too (with recursion limit)
                    sub_chunks = self._split_large_chunk(
                        final_text,
                        f"{chunk_id}_part_{chunk_index}",
                        metadata.copy(),
                        depth + 1
                    )
                    for sub_chunk in sub_chunks:
                        sub_chunk.chunk_index = chunk_index
                        chunks.append(sub_chunk)
                        chunk_index += 1
                else:
                    chunk = ContiguousChunk(
                        id=f"{chunk_id}_part_{chunk_index}" if chunk_index > 0 else chunk_id,
                        text=final_text,
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
        Merge small chunks (<merge_threshold tokens) with next siblings.
        Only merges if same parent and element type.
        Ensures merged chunk doesn't exceed max_tokens.
        """
        if start_idx >= len(chunks):
            return [], start_idx
        
        first_chunk = chunks[start_idx]
        parent_id = first_chunk.get('parent_id')
        element_type = first_chunk.get('element_type', '')
        
        # Only merge paragraph/subparagraph level chunks
        if element_type not in ['paragraph', 'subparagraph']:
            return [], start_idx
        
        merged_parts = []
        merged_metadata = first_chunk.get('metadata', {}).copy()
        total_tokens = 0
        current_idx = start_idx
        chunk_ids = []
        children_ids = []
        
        while current_idx < len(chunks):
            chunk = chunks[current_idx]
            
            # Only merge if same parent and element type
            if chunk.get('parent_id') != parent_id or chunk.get('element_type', '') != element_type:
                break
            
            chunk_text = self._clean_text(chunk.get('text', ''))
            chunk_tokens = self.count_tokens(chunk_text)
            
            # Stop if adding this would exceed max_tokens (and we already have min_tokens)
            if total_tokens + chunk_tokens > self.max_tokens and total_tokens >= self.min_tokens:
                break
            
            merged_parts.append(chunk_text)
            total_tokens += chunk_tokens
            chunk_ids.append(chunk.get('id'))
            children_ids.extend(chunk.get('children_ids', []))
            current_idx += 1
            
            # Stop if we've reached target size
            if total_tokens >= self.min_tokens:
                break
        
        # Only merge if we actually merged multiple chunks
        if current_idx <= start_idx + 1:
            return [], start_idx
        
        # Combine text with single newline (not double)
        merged_text = '\n'.join(merged_parts)
        merged_text = self._clean_text(merged_text)
        
        # Verify merged chunk is within limits
        final_tokens = self.count_tokens(merged_text)
        if final_tokens > self.max_tokens:
            # Merged chunk is too large - split it
            return self._split_large_chunk(merged_text, chunk_ids[0], merged_metadata), current_idx
        
        merged_chunk = ContiguousChunk(
            id=chunk_ids[0],
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
        max_chunk_size: Optional[int] = None  # Deprecated
    ) -> List[ContiguousChunk]:
        """
        Process chunks using structure-first chunking strategy.
        
        Strategy:
        1. Clean text (remove unnecessary whitespace)
        2. Chunk by structure first (paragraph/subparagraph boundaries)
        3. If <merge_threshold tokens, merge with next sibling until ~min_tokens
        4. If >split_threshold tokens, split by sub-items
        5. Target: 300-700 tokens per chunk
        
        Args:
            chunks: List of chunk dictionaries from XML parser
            max_chunk_size: Deprecated
            
        Returns:
            List of ContiguousChunk objects
        """
        result_chunks = []
        chunk_id_map = {}
        
        i = 0
        while i < len(chunks):
            chunk_dict = chunks[i]
            original_id = chunk_dict.get('id')
            raw_text = chunk_dict.get('text', '')
            
            # Clean text first
            text = self._clean_text(raw_text)
            
            parent_id = chunk_dict.get('parent_id')
            children_ids = chunk_dict.get('children_ids', [])
            metadata = chunk_dict.get('metadata', {}).copy()
            
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
                if element_type in ['paragraph', 'subparagraph']:
                    merged_chunks, next_idx = self._merge_small_chunks(chunks, i)
                    if next_idx > i + 1:
                        # Successfully merged
                        result_chunks.extend(merged_chunks)
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
                # Large chunk: split by legal boundaries
                split_chunks = self._split_large_chunk(text, original_id, metadata)
                
                if len(split_chunks) > 1:
                    # Successfully split
                    split_chunks[0].parent_id = parent_id
                    split_chunks[0].children_ids = children_ids.copy()
                    split_chunks[0].chunk_index = None
                    
                    new_ids = [original_id]
                    for j, split_chunk in enumerate(split_chunks[1:], 1):
                        split_chunk.parent_id = original_id
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
                chunk.parent_id = chunk_id_map[chunk.parent_id][0]
            
            updated_children = []
            for child_id in chunk.children_ids:
                if child_id in chunk_id_map:
                    updated_children.extend(chunk_id_map[child_id])
                else:
                    updated_children.append(child_id)
            chunk.children_ids = updated_children
        
        return result_chunks


def chunk_for_rag_contiguous(
    chunks: List[Dict], 
    chunk_size: int = 500,
    chunk_overlap: int = 50
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
    min_tokens = max(300, int(chunk_size * 0.6))
    max_tokens = min(700, int(chunk_size * 1.4))
    
    chunker = StructureFirstChunker(
        target_tokens=chunk_size,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        split_threshold=max_tokens,  # Split if >max_tokens (not 1000!)
        merge_threshold=200,
        chunk_overlap=chunk_overlap
    )
    
    text_chunks = chunker.chunk_with_relationships(chunks)
    
    # Convert to dictionaries
    rag_chunks = []
    for chunk in text_chunks:
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
                'token_count': token_count,
            }
        }
        rag_chunks.append(rag_chunk)
    
    return rag_chunks
