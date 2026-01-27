"""
Analyze chunk statistics from JSON chunk files.
"""
import json
import tiktoken
from pathlib import Path
from typing import List, Dict
from statistics import median


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens using tiktoken."""
    try:
        encoder = tiktoken.encoding_for_model(model)
    except KeyError:
        encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


def analyze_chunks(chunks_path: str) -> Dict:
    """Analyze chunks and return statistics."""
    print(f"Loading chunks from {chunks_path}...")
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks. Calculating token counts...")
    
    token_counts = []
    char_counts = []
    chunks_with_tokens = []
    
    for chunk in chunks:
        text = chunk.get('text', '')
        char_count = len(text)
        char_counts.append(char_count)
        
        # Check if token_count already exists in metadata
        token_count = None
        if 'metadata' in chunk and isinstance(chunk['metadata'], dict):
            token_count = chunk['metadata'].get('token_count')
        
        # Calculate if not present
        if token_count is None:
            token_count = count_tokens(text)
        
        token_counts.append(token_count)
        chunks_with_tokens.append({
            'id': chunk.get('id', 'unknown'),
            'token_count': token_count,
            'char_count': char_count,
            'text': text[:200] + '...' if len(text) > 200 else text
        })
    
    # Calculate statistics
    total_chunks = len(chunks)
    avg_tokens = sum(token_counts) / total_chunks if total_chunks > 0 else 0
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    median_tokens = median(token_counts) if token_counts else 0
    
    avg_chars = sum(char_counts) / total_chunks if total_chunks > 0 else 0
    chars_to_tokens_ratio = avg_chars / avg_tokens if avg_tokens > 0 else 0
    
    # Distribution analysis
    sweet_spot = sum(1 for tc in token_counts if 300 <= tc <= 700)
    below_300 = sum(1 for tc in token_counts if tc < 300)
    above_700 = sum(1 for tc in token_counts if tc > 700)
    
    # Percentiles
    sorted_tokens = sorted(token_counts)
    p25_idx = int(len(sorted_tokens) * 0.25)
    p50_idx = int(len(sorted_tokens) * 0.50)
    p75_idx = int(len(sorted_tokens) * 0.75)
    p90_idx = int(len(sorted_tokens) * 0.90)
    p95_idx = int(len(sorted_tokens) * 0.95)
    p99_idx = int(len(sorted_tokens) * 0.99)
    
    percentiles = {
        '25th': sorted_tokens[p25_idx] if sorted_tokens else 0,
        '50th': sorted_tokens[p50_idx] if sorted_tokens else 0,
        '75th': sorted_tokens[p75_idx] if sorted_tokens else 0,
        '90th': sorted_tokens[p90_idx] if sorted_tokens else 0,
        '95th': sorted_tokens[p95_idx] if sorted_tokens else 0,
        '99th': sorted_tokens[p99_idx] if sorted_tokens else 0,
    }
    
    # Distribution buckets
    buckets = {
        '0-300': sum(1 for tc in token_counts if 0 <= tc < 300),
        '300-500': sum(1 for tc in token_counts if 300 <= tc < 500),
        '500-700': sum(1 for tc in token_counts if 500 <= tc < 700),
        '700-1000': sum(1 for tc in token_counts if 700 <= tc < 1000),
        '1000+': sum(1 for tc in token_counts if tc >= 1000),
    }
    
    return {
        'total_chunks': total_chunks,
        'avg_tokens': avg_tokens,
        'min_tokens': min_tokens,
        'max_tokens': max_tokens,
        'median_tokens': median_tokens,
        'avg_chars': avg_chars,
        'chars_to_tokens_ratio': chars_to_tokens_ratio,
        'sweet_spot': sweet_spot,
        'below_300': below_300,
        'above_700': above_700,
        'percentiles': percentiles,
        'buckets': buckets,
        'chunks_with_tokens': chunks_with_tokens,
    }


def print_statistics(stats: Dict):
    """Print formatted statistics."""
    print("\n" + "="*60)
    print("CHUNK STATISTICS")
    print("="*60)
    
    print(f"\n**Total Chunks**: {stats['total_chunks']:,}")
    print(f"**Average Tokens**: {stats['avg_tokens']:.1f}")
    print(f"**Min Tokens**: {stats['min_tokens']:,}")
    print(f"**Max Tokens**: {stats['max_tokens']:,}")
    print(f"**Median Tokens**: {stats['median_tokens']:.1f}")
    
    print(f"\n**Average Characters**: {stats['avg_chars']:.1f}")
    print(f"**Characters-to-Tokens Ratio**: {stats['chars_to_tokens_ratio']:.2f}")
    
    print("\n### Distribution")
    total = stats['total_chunks']
    sweet_pct = (stats['sweet_spot'] / total * 100) if total > 0 else 0
    below_pct = (stats['below_300'] / total * 100) if total > 0 else 0
    above_pct = (stats['above_700'] / total * 100) if total > 0 else 0
    
    print(f"- Sweet spot (300-700 tokens): {stats['sweet_spot']:,} chunks ({sweet_pct:.1f}%)")
    print(f"- Below 300 tokens: {stats['below_300']:,} chunks ({below_pct:.1f}%)")
    print(f"- Above 700 tokens: {stats['above_700']:,} chunks ({above_pct:.1f}%)")
    
    print("\n### Percentiles")
    p = stats['percentiles']
    print(f"- 25th percentile: {p['25th']:,} tokens")
    print(f"- 50th percentile (median): {p['50th']:,} tokens")
    print(f"- 75th percentile: {p['75th']:,} tokens")
    print(f"- 90th percentile: {p['90th']:,} tokens")
    print(f"- 95th percentile: {p['95th']:,} tokens")
    print(f"- 99th percentile: {p['99th']:,} tokens")
    
    print("\n### Distribution Buckets")
    b = stats['buckets']
    for bucket, count in b.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"- {bucket} tokens: {count:,} chunks ({pct:.1f}%)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    chunks_path = data_dir / "chunks.json"
    
    stats = analyze_chunks(str(chunks_path))
    print_statistics(stats)
    
    # Show some examples of large chunks
    large_chunks = [c for c in stats['chunks_with_tokens'] if c['token_count'] > 700]
    if large_chunks:
        print(f"\n### Sample Chunks Exceeding 700 Tokens (showing first 5)")
        print("="*60)
        for i, chunk in enumerate(large_chunks[:5], 1):
            print(f"\n{i}. Chunk ID: {chunk['id']}")
            print(f"   Tokens: {chunk['token_count']:,}")
            print(f"   Characters: {chunk['char_count']:,}")
            print(f"   Preview: {chunk['text'][:150]}...")
