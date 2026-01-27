---
name: analyze-chunks
description: Analyzes JSON chunk files to compute statistics including average/max/min chunk sizes, token distributions, and identifies chunks exceeding specified thresholds. Use when the user asks for chunk statistics, wants to analyze chunk files, needs distribution analysis, or wants to find chunks above a certain token count.
---

# Analyze Chunks

Analyzes JSON chunk files and provides comprehensive statistics about chunk sizes, token counts, and distributions.

## Quick Start

When analyzing chunks:

1. Load the JSON file containing chunks
2. Calculate token counts for each chunk (if not already present)
3. Compute statistics based on user's request (general overview or specific metrics)
4. Format results clearly

## Chunk File Structure

Chunks are JSON arrays where each chunk has:
- `id`: Unique identifier
- `text`: The chunk content (required for token counting)
- `metadata`: May contain `token_count` if already calculated
- Other fields: `element_type`, `identifier`, `parent_id`, `children_ids`, etc.

## Token Counting

Use `tiktoken` for accurate token counting (same as `chunk2.py`):

```python
import tiktoken
import json

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens using tiktoken."""
    try:
        encoder = tiktoken.encoding_for_model(model)
    except KeyError:
        encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))
```

If `metadata.token_count` exists, use it. Otherwise, calculate from `text`.

## General Statistics

When user asks for "general picture" or "overview", provide:

1. **Basic Stats**:
   - Total number of chunks
   - Average tokens per chunk
   - Min tokens
   - Max tokens
   - Median tokens (if helpful)

2. **Distribution Analysis**:
   - Chunks in sweet spot (300-700 tokens) - count and percentage
   - Chunks below 300 tokens - count and percentage
   - Chunks above 700 tokens - count and percentage
   - Optional: Histogram or bucket distribution (0-300, 300-500, 500-700, 700-1000, 1000+)

3. **Character vs Token Stats**:
   - Average characters per chunk
   - Characters-to-tokens ratio

## Specific Statistics

When user asks for specific stats:

- **"Chunks >X tokens"**: List chunks with token count > X, including:
  - Chunk ID
  - Token count
  - Character count
  - Text preview (first 100-200 chars)
  - Optional: Full text if small enough

- **"Distribution"**: Provide detailed distribution breakdown:
  - Token count ranges with counts
  - Percentiles (25th, 50th, 75th, 90th, 95th, 99th)
  - Visual representation if helpful (text-based histogram)

- **"Average/Max/Min"**: Just the requested metrics

## Output Format

Format results clearly:

```markdown
## Chunk Statistics

**Total Chunks**: X
**Average Tokens**: X.X
**Min Tokens**: X
**Max Tokens**: X
**Median Tokens**: X

### Distribution
- Sweet spot (300-700 tokens): X chunks (X.X%)
- Below 300 tokens: X chunks (X.X%)
- Above 700 tokens: X chunks (X.X%)

### Chunks Exceeding Threshold (>X tokens)
[List of chunks with details]
```

## Implementation Notes

1. Handle large files efficiently - read JSON in chunks if needed
2. Cache token counts in memory to avoid recalculating
3. For very large files, consider sampling or streaming
4. If token_count already exists in metadata, prefer it over recalculating
5. Handle edge cases: empty chunks, missing text fields, invalid JSON

## Examples

**User**: "Give me stats on chunks.json"
**Response**: Full general statistics overview

**User**: "Show me chunks over 1000 tokens"
**Response**: List of chunks with token_count > 1000

**User**: "What's the distribution of chunk sizes?"
**Response**: Detailed distribution breakdown with buckets and percentages
