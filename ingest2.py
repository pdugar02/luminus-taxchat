"""
Alternative XML Parser using contiguous chunking with overlap.
Uses chunk2.py for chunking that keeps contiguous text together.
"""

import json
import argparse
from dataclasses import asdict
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


def main(chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Main function to parse XML and extract chunks using structure-first chunking.
    
    Args:
        chunk_size: Target size for chunks (in tokens). Default is 500 (sweet spot 300-700).
        chunk_overlap: Overlap between chunks (in tokens). Default is 50 (range 0-80).
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
    
    # Apply structure-first chunking for RAG (with token counting)
    print(f"\nApplying structure-first chunking for RAG (chunk_size={chunk_size} tokens, chunk_overlap={chunk_overlap} tokens)...")
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
    print("\n=== Sample RAG Chunks (Structure-First with Token Counting) ===")
    for i, chunk in enumerate(rag_chunks[:5]):
        print(f"\nRAG Chunk {i+1}:")
        print(f"  ID: {chunk['id']}")
        token_count = chunk['metadata'].get('token_count', 'N/A')
        print(f"  Token count: {token_count}")
        print(f"  Text length: {len(chunk['text'])} chars")
        if chunk['metadata'].get('start_char') is not None:
            print(f"  Position: {chunk['metadata']['start_char']}-{chunk['metadata']['end_char']}")
        print(f"  Text preview: {chunk['text'][:100]}...")
    
    # Show token statistics
    print("\n=== Token Statistics ===")
    token_counts = [chunk['metadata'].get('token_count', 0) for chunk in rag_chunks if chunk['metadata'].get('token_count')]
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        print(f"Total chunks: {len(rag_chunks)}")
        print(f"Average tokens per chunk: {avg_tokens:.1f}")
        print(f"Min tokens: {min_tokens}")
        print(f"Max tokens: {max_tokens}")
        
        # Count chunks in sweet spot (300-700 tokens)
        sweet_spot_count = sum(1 for tc in token_counts if 300 <= tc <= 700)
        print(f"Chunks in sweet spot (300-700 tokens): {sweet_spot_count} ({100*sweet_spot_count/len(token_counts):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse US Code Title 26 XML with structure-first chunking")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target size for chunks in tokens (default: 500, sweet spot 300-700)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks in tokens (default: 50, range 0-80)"
    )
    
    args = parser.parse_args()
    main(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

