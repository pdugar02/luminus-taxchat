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
        chunk_overlap: int = 50,
    ):
        """
        Initialize chunker.
        """
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.split_threshold = split_threshold
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoder.encode(text))
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing unnecessary whitespace and normalizing.
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

    def _split_at_subsections(self, text: str, chunk_id: str, metadata: Dict) -> List[ContiguousChunk]:
        """
        Split section text into chunks by subsection boundaries, but only when needed.

        Behavior:
        - Extract the section intro (everything before the first subsection marker).
        - If the overall text is too large, split it on the subsection boundary that produces the
          most balanced halves (closest to the midpoint by token count), then recurse.
        - Stop splitting when the chunk is <= split_threshold OR when splitting would require
          going smaller than a single subsection.
        - The intro text is prepended to every returned chunk for context.
        """
        boundaries = self._find_subsection_boundaries(text)
        if not boundaries:
            return []
        first_pos, _ = boundaries[0]
        intro_text = text[:first_pos].strip()
        subsections: List[Tuple[int, int, str]] = []
        for i, (pos, _label) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            subsections.append((pos, end, text[pos:end].strip()))

        # build text snippet from subsections i to j
        def build_text(i: int, j: int) -> str:
            body = " ".join(text for pos, end, text in subsections[i:j] if text)
            return f"{intro_text} {body}".strip() if intro_text else body

        def split_ranges(i: int, j: int) -> List[Tuple[int, int]]:
            if j - i <= 1:
                return [(i, j)]
            if self.count_tokens(build_text(i, j)) <= self.split_threshold:
                return [(i, j)]
            best_k = None
            best_diff = None
            for k in range(i + 1, j):
                left = self.count_tokens(build_text(i, k))
                right = self.count_tokens(build_text(k, j))
                diff = abs(left - right)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_k = k
            if best_k is None:
                return [(i, j)]
            return split_ranges(i, best_k) + split_ranges(best_k, j)

        ranges = split_ranges(0, len(subsections))
        out: List[ContiguousChunk] = []
        for idx, (i, j) in enumerate(ranges, 1):
            start_pos = subsections[i][0]
            end_pos = subsections[j - 1][1]
            out.append(
                ContiguousChunk(
                    id=f"{chunk_id}_part_{idx}",
                    text=build_text(i, j),
                    metadata=metadata.copy(),
                    chunk_index=idx,
                    start_char=start_pos,
                    end_char=end_pos,
                )
            )
        return out
    
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

        # Prefer splitting at subsection boundaries (balanced + recursive), and stop at single-subsection granularity.
        subsection_chunks = self._split_at_subsections(text, chunk_id, metadata)
        if subsection_chunks:
            # If it didn't actually split (single output chunk), keep original ID semantics.
            if len(subsection_chunks) == 1:
                subsection_chunks[0].id = chunk_id
                subsection_chunks[0].chunk_index = None
                subsection_chunks[0].start_char = 0
                subsection_chunks[0].end_char = len(text)
            return subsection_chunks

        # If no subsection markers are found, do not attempt deeper splitting in this phase.
        # (For US tax code sections, subsection markers are expected to exist.)
        return [ContiguousChunk(
            id=chunk_id,
            text=text,
            metadata=metadata,
            chunk_index=None,
            start_char=0,
            end_char=len(text)
        )]

    def _section_key(self, identifier: Optional[str]) -> Optional[str]:
        """
        Return the second-to-last character of the zero-padded identifier (integer string).
        Used to decide if two consecutive chunks are in the same "section" for merging.
        E.g. "4"/"5" -> "04"/"05" -> key "0" (merge); "68"/"71" -> "6"/"7" (do not merge).
        Returns None if identifier is missing or length < 2 after padding.
        """
        if identifier is None:
            return None
        s = str(identifier).strip()
        if not s:
            return None
        normalized = s.zfill(2)  # at least two chars: "4" -> "04"
        if len(normalized) < 2:
            return None
        return normalized[-2]

    def _merge_small_chunks(
        self, chunks: List[Dict], start_idx: int
    ) -> Tuple[List[ContiguousChunk], int]:
        """
        Merge small chunks (<min_tokens) with next consecutive chunks that share
        the same section key (second-to-last digit of zero-padded identifier).
        Stops when merged token count > max_tokens or no more same-section consecutive chunks.
        Returns (list of ContiguousChunk(s), next index to process).
        """
        if start_idx >= len(chunks):
            return [], start_idx

        first_chunk = chunks[start_idx]
        section_key = self._section_key(first_chunk.get('identifier'))
        if section_key is None:
            # No merge: emit single chunk
            c = first_chunk
            text = self._clean_text(c.get('text', ''))
            single = ContiguousChunk(
                id=c.get('id'),
                text=text,
                parent_id=c.get('parent_id'),
                children_ids=list(c.get('children_ids', [])),
                metadata=c.get('metadata', {}).copy(),
                chunk_index=None,
                start_char=0,
                end_char=len(text),
            )
            single.metadata['tag'] = c.get('element_type', '')
            single.metadata['identifier'] = c.get('identifier')
            single.metadata['identifiers'] = [c.get('identifier')] if c.get('identifier') is not None else []
            single.metadata['heading'] = single.metadata.get('heading', '')
            single.metadata['element_id'] = c.get('id')
            return [single], start_idx + 1

        merged_parts = []
        merged_metadata = first_chunk.get('metadata', {}).copy()
        merged_metadata['tag'] = first_chunk.get('element_type', '')
        merged_metadata['identifier'] = first_chunk.get('identifier')
        merged_metadata['heading'] = merged_metadata.get('heading', '')
        merged_metadata['element_id'] = first_chunk.get('id')
        total_tokens = 0
        chunk_ids = []
        children_ids = []
        section_identifiers = []  # list of section identifiers for merged chunk
        current_idx = start_idx

        while current_idx < len(chunks):
            chunk = chunks[current_idx]
            if self._section_key(chunk.get('identifier')) != section_key:
                break
            chunk_text = self._clean_text(chunk.get('text', ''))
            chunk_tokens = self.count_tokens(chunk_text)
            if total_tokens + chunk_tokens > self.max_tokens and total_tokens >= self.min_tokens:
                break
            section_identifiers.append(chunk.get('identifier'))
            merged_parts.append(chunk_text)
            total_tokens += chunk_tokens
            chunk_ids.append(chunk.get('id'))
            children_ids.extend(chunk.get('children_ids', []))
            current_idx += 1
            if total_tokens > self.max_tokens:
                break

        merged_metadata['identifiers'] = section_identifiers

        if current_idx <= start_idx + 1:
            c = first_chunk
            text = self._clean_text(c.get('text', ''))
            single = ContiguousChunk(
                id=c.get('id'),
                text=text,
                parent_id=c.get('parent_id'),
                children_ids=list(c.get('children_ids', [])),
                metadata=merged_metadata,
                chunk_index=None,
                start_char=0,
                end_char=len(text),
            )
            return [single], start_idx + 1

        merged_text = '\n'.join(merged_parts)
        merged_text = self._clean_text(merged_text)
        final_tokens = self.count_tokens(merged_text)
        if final_tokens > self.max_tokens:
            split_chunks = self._split_large_chunk(merged_text, chunk_ids[0], merged_metadata)
            return split_chunks, current_idx

        merged_chunk = ContiguousChunk(
            id=chunk_ids[0],
            text=merged_text,
            parent_id=first_chunk.get('parent_id'),
            children_ids=children_ids,
            metadata=merged_metadata,
            chunk_index=None,
            start_char=0,
            end_char=len(merged_text),
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
        3. If >split_threshold tokens, split (prefers subsection boundaries)
        4. Target: 300-700 tokens per chunk
        
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
            
            # Add element type and identifier to metadata (identifiers = list for single or merged sections)
            metadata['tag'] = chunk_dict.get('element_type', '')
            metadata['identifier'] = chunk_dict.get('identifier')
            metadata['identifiers'] = [chunk_dict.get('identifier')] if chunk_dict.get('identifier') is not None else []
            metadata['heading'] = metadata.get('heading', '')
            metadata['element_id'] = chunk_dict.get('id')
            
            token_count = self.count_tokens(text)
            
            # Strategy: only split when too large; otherwise keep as a single chunk
            if token_count > self.split_threshold:
                # Large chunk: split (prefers subsection boundaries inside _split_large_chunk)
                split_chunks = self._split_large_chunk(text, original_id, metadata)
                
                if len(split_chunks) > 1:
                    # Successfully split
                    split_chunks[0].parent_id = parent_id
                    split_chunks[0].children_ids = children_ids.copy()
                    split_chunks[0].chunk_index = None
                    
                    # Map original_id -> actual produced chunk IDs
                    new_ids = [split_chunks[0].id]
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
            elif token_count < self.min_tokens:
                merged_list, next_i = self._merge_small_chunks(chunks, i)
                merged_ids = [c.id for c in merged_list]
                for idx in range(i, next_i):
                    oid = chunks[idx].get('id')
                    if oid is not None:
                        chunk_id_map[oid] = merged_ids
                result_chunks.extend(merged_list)
                i = next_i
            else:
                # Chunk is in the sweet spot: keep as single chunk
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
        split_threshold=max_tokens,  # Split if >max_tokens
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
