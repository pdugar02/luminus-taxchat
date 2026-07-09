#!/usr/bin/env python3
"""
Analyze rag_chunks2.json and flag chunks with abnormally large text contents.
Uses character count; flags outliers via mean + 2*std and top percentiles.
"""

import json
import statistics
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
CHUNKS_PATH = DATA_DIR / "rag_chunks2.json"
OUTPUT_PATH = DATA_DIR / "large_chunks_report.txt"


def main():
    print("Loading rag_chunks2.json...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("No chunks found.")
        return

    # Collect text lengths (characters) and map to chunk id for reporting
    lengths = []
    id_to_len = {}
    id_to_chunk = {}
    for ch in chunks:
        text = ch.get("text", "")
        n = len(text)
        lengths.append(n)
        id_to_len[ch["id"]] = n
        id_to_chunk[ch["id"]] = ch

    n_chunks = len(lengths)
    mean_len = statistics.mean(lengths)
    try:
        stdev_len = statistics.stdev(lengths)
    except statistics.StatisticsError:
        stdev_len = 0.0
    median_len = statistics.median(lengths)

    # Percentiles (approximate via sorted)
    sorted_lens = sorted(lengths)
    p95 = sorted_lens[int(0.95 * n_chunks)] if n_chunks else 0
    p99 = sorted_lens[int(0.99 * n_chunks)] if n_chunks else 0

    threshold_2std = mean_len + 2 * stdev_len if stdev_len else mean_len
    # Also consider "abnormal" as top 1% by length
    threshold_p99 = p99

    print("\n--- Text length (characters) statistics ---")
    print(f"Chunks: {n_chunks}")
    print(f"Mean:   {mean_len:.0f}")
    print(f"Median: {median_len:.0f}")
    print(f"Std:    {stdev_len:.0f}")
    print(f"Min:    {min(lengths)}")
    print(f"Max:    {max(lengths)}")
    print(f"95th %: {p95}")
    print(f"99th %: {p99}")
    print(f"Mean + 2*std: {threshold_2std:.0f}")

    # Flag: abnormal = above mean+2*std OR above 99th percentile (whichever catches outliers)
    use_threshold = max(threshold_2std, threshold_p99) if stdev_len > 0 else threshold_p99
    flagged = [
        (cid, id_to_len[cid], id_to_chunk[cid])
        for cid in id_to_len
        if id_to_len[cid] >= use_threshold
    ]
    flagged.sort(key=lambda x: -x[1])

    print(f"\n--- Flagged chunks (length >= {use_threshold:.0f}) ---")
    print(f"Count: {len(flagged)}")

    lines = []
    lines.append("RAG chunks with abnormally large text (rag_chunks2.json)\n")
    lines.append("=" * 60 + "\n")
    lines.append(f"Threshold: >= {use_threshold:.0f} characters (mean + 2*std or 99th %)\n")
    lines.append(f"Stats: mean={mean_len:.0f}, median={median_len:.0f}, std={stdev_len:.0f}\n")
    lines.append(f"Flagged: {len(flagged)} chunks\n\n")

    for cid, length, ch in flagged:
        meta = ch.get("metadata", {})
        heading = meta.get("heading", "")
        token_count = meta.get("token_count", "?")
        preview = (ch.get("text", "")[:200] + "…") if len(ch.get("text", "")) > 200 else ch.get("text", "")
        lines.append(f"id: {cid}\n")
        lines.append(f"  length: {length} chars, token_count: {token_count}\n")
        lines.append(f"  heading: {heading}\n")
        lines.append(f"  preview: {preview}\n\n")
        print(f"  {cid}: {length} chars (tokens: {token_count}) — {heading or '(no heading)'}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        out.writelines(lines)
    print(f"\nReport written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
