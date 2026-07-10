"""
Legacy standalone ingestion script (superseded by ingest.py at the repo root).

Kept only for reference: it parses usc26.xml with XMLParser and runs the old
character-based chunk_for_rag. The live pipeline (ingest.py) imports XMLParser
from xml_parser.py and chunks with chunk_for_rag_contiguous in chunk.py instead.
Nothing in the running system imports this module.
"""

import json
import argparse
from dataclasses import asdict
from pathlib import Path

from xml_parser import XMLParser
from old_scripts.old_chunk import chunk_for_rag


def main(chunk_size: int = 2000, chunk_overlap: int = 400):
    """
    Main function to parse XML and extract chunks.

    Args:
        chunk_size: Maximum size for chunks (in characters). Default is 1000.
        chunk_overlap: Overlap between chunks (in characters). Default is 200.
                      Set to 0 to disable overlap.
    """
    data_dir = Path(__file__).parent.parent / "data"
    xml_path = data_dir / "usc26.xml"
    output_path = data_dir / "chunks.json"
    rag_output_path = data_dir / "rag_chunks.json"

    # Parse XML
    parser = XMLParser(str(xml_path))
    chunks = parser.parse()

    # Save raw chunks
    parser.save_chunks(str(output_path))

    # Convert to dictionaries for chunking
    chunks_dict = [asdict(chunk) for chunk in chunks]

    # Apply chunking for RAG (split large chunks)
    print(f"\nApplying chunking for RAG (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    rag_chunks = chunk_for_rag(chunks_dict, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Save RAG-ready chunks
    with open(rag_output_path, 'w', encoding='utf-8') as f:
        json.dump(rag_chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(rag_chunks)} RAG-ready chunks to {rag_output_path}")

    # Print summary
    print("\n=== Parsing Summary ===")
    print(f"Total raw chunks: {len(chunks)}")
    print(f"Total RAG chunks: {len(rag_chunks)}")

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
        print(f"  Parent ID: {chunk.parent_id}")
        print(f"  Children: {len(chunk.children_ids)}")
        # Show hierarchy chain
        hierarchy = chunk.metadata.get('hierarchy', {})
        if hierarchy:
            print("  Hierarchy:")
            for level, info in hierarchy.items():
                print(f"    {level}: {info.get('identifier')} - {info.get('heading')}")
        print(f"  Text preview: {chunk.text[:100]}...")

    print("\n=== Sample RAG Chunks ===")
    for i, chunk in enumerate(rag_chunks[:3]):
        print(f"\nRAG Chunk {i+1}:")
        print(f"  ID: {chunk['id']}")
        print(f"  Parent ID: {chunk['metadata'].get('parent_id')}")
        print(f"  Children: {len(chunk['metadata'].get('children_ids', []))}")
        # Show hierarchy chain
        hierarchy = chunk['metadata'].get('hierarchy', {})
        if hierarchy:
            print("  Hierarchy:")
            for level, info in hierarchy.items():
                print(f"    {level}: {info.get('identifier')} - {info.get('heading')}")
        print(f"  Text preview: {chunk['text'][:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse US Code Title 26 XML and extract chunks for RAG')
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=2000,
        help='Maximum chunk size in characters (default: 1000)'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=400,
        help='Overlap between chunks in characters (default: 200). Set to 0 to disable overlap.'
    )
    args = parser.parse_args()
    main(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
