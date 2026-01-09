"""
Alternative XML Parser using contiguous chunking with overlap.
Uses chunk2.py for chunking that keeps contiguous text together.
"""

from lxml import etree
from typing import List, Dict, Optional, Any
import json
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from chunk2 import chunk_for_rag_contiguous

# Import the XMLParser class from the original ingest.py
# We'll reuse the same parsing logic
import sys
import importlib.util

# Load the XMLParser from ingest.py
spec = importlib.util.spec_from_file_location("ingest", "ingest.py")
ingest_module = importlib.util.module_from_spec(spec)
sys.modules["ingest"] = ingest_module
spec.loader.exec_module(ingest_module)

XMLParser = ingest_module.XMLParser


def main(chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Main function to parse XML and extract chunks using contiguous chunking.
    
    Args:
        chunk_size: Maximum size for chunks (in characters). Default is 1000.
        chunk_overlap: Overlap between chunks (in characters). Default is 200.
                      Set to 0 to disable overlap.
    """
    data_dir = Path(__file__).parent / "data"
    xml_path = data_dir / "usc26.xml"
    output_path = data_dir / "chunks2.json"
    rag_output_path = data_dir / "rag_chunks2.json"
    
    # Parse XML (reuse the same parser)
    parser = XMLParser(str(xml_path))
    chunks = parser.parse()
    
    # Save raw chunks
    parser.save_chunks(str(output_path))
    
    # Convert to dictionaries for chunking
    chunks_dict = [asdict(chunk) for chunk in chunks]
    
    # Apply contiguous chunking for RAG (with overlap)
    print(f"\nApplying contiguous chunking for RAG (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    rag_chunks = chunk_for_rag_contiguous(chunks_dict, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Save RAG-ready chunks
    with open(rag_output_path, 'w', encoding='utf-8') as f:
        json.dump(rag_chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(rag_chunks)} RAG-ready chunks to {rag_output_path}")
    
    # Print summary
    print("\n=== Parsing Summary ===")
    print(f"Total raw chunks: {len(chunks)}")
    print(f"Total RAG chunks (contiguous with overlap): {len(rag_chunks)}")
    
    # Count by type
    type_counts = {}
    for chunk in chunks:
        type_counts[chunk.element_type] = type_counts.get(chunk.element_type, 0) + 1
    
    print("\nChunks by type:")
    for elem_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {elem_type}: {count}")
    
    # Show sample chunks
    print("\n=== Sample Raw Chunks ===")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.id}")
        print(f"  Type: {chunk.element_type}")
        print(f"  Identifier: {chunk.identifier}")
        print(f"  Text length: {len(chunk.text)}")
        print(f"  Text preview: {chunk.text[:100]}...")
    
    # Show sample RAG chunks
    print("\n=== Sample RAG Chunks (Contiguous with Overlap) ===")
    for i, chunk in enumerate(rag_chunks[:5]):
        print(f"\nRAG Chunk {i+1}:")
        print(f"  ID: {chunk['id']}")
        print(f"  Text length: {len(chunk['text'])}")
        if chunk['metadata'].get('start_char') is not None:
            print(f"  Position: {chunk['metadata']['start_char']}-{chunk['metadata']['end_char']}")
        print(f"  Text preview: {chunk['text'][:100]}...")
    
    # Show overlap statistics
    print("\n=== Overlap Statistics ===")
    overlap_count = 0
    total_overlap = 0
    for chunk in rag_chunks:
        if chunk['metadata'].get('chunk_index') is not None and chunk['metadata']['chunk_index'] > 0:
            overlap_count += 1
            # Estimate overlap from position
            if chunk['metadata'].get('start_char') is not None:
                prev_end = chunk['metadata'].get('start_char', 0)
                # Rough estimate: overlap is roughly chunk_overlap
                total_overlap += chunk_overlap
    
    print(f"Chunks with overlap: {overlap_count}")
    if overlap_count > 0:
        avg_overlap = total_overlap / overlap_count if overlap_count > 0 else 0
        print(f"Average overlap: ~{avg_overlap:.0f} characters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse US Code Title 26 XML with contiguous chunking")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum size for chunks in characters (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)"
    )
    
    args = parser.parse_args()
    main(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

