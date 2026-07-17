#!/usr/bin/env python3
"""
Retrieval evaluation against the golden set (eval/golden_set.json).

Measures whether the expected IRC sections appear in the retrieved chunks —
separately from answer quality, so retrieval regressions are visible before
any prompt tuning.

Usage:
    python scripts/eval_retrieval.py               # fast: plain hybrid retrieve, no LLM
    python scripts/eval_retrieval.py --full        # full pipeline (condense/classify/expand), needs LLM
    python scripts/eval_retrieval.py --top-k 10
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from query import get_rag, handle_query  # noqa: E402


def _bare_section(identifier: str) -> str | None:
    m = re.match(r'\d+[A-Z]{0,2}(?:-\d+)?', str(identifier))
    return m.group(0) if m else None


def _retrieved_sections(sources: list[dict]) -> set[str]:
    found = set()
    for s in sources:
        meta = s.get("metadata", {})
        idents = meta.get("identifiers") or ([meta["identifier"]] if meta.get("identifier") else [])
        for ident in idents:
            bare = _bare_section(ident)
            if bare:
                found.add(bare)
    return found


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval recall against the golden set")
    parser.add_argument("--full", action="store_true",
                        help="Run the full query pipeline (classify + expand); default is plain hybrid retrieve")
    parser.add_argument("--top-k", type=int, default=8, help="top_k for plain retrieve mode (default 8)")
    parser.add_argument("--golden", default=str(Path(__file__).parent.parent / "eval" / "golden_set.json"))
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text())
    rag = get_rag()

    total_expected = 0
    total_found = 0
    full_hits = 0
    misses = []

    for item in golden:
        question = item["question"]
        expected = set(item["expected_sections"])

        if args.full:
            payload, _ = handle_query({"question": question, "retrieve_only": True})
            sources = payload.get("sources", [])
        else:
            sources = rag.retrieve(question, top_k=args.top_k)

        found_sections = _retrieved_sections(sources)
        hit = expected & found_sections
        total_expected += len(expected)
        total_found += len(hit)
        if hit == expected:
            full_hits += 1
            status = "PASS"
        else:
            misses.append((question, sorted(expected - found_sections), sorted(found_sections)))
            status = "MISS"
        print(f"[{status}] {question}")
        print(f"       expected {sorted(expected)}  found {sorted(hit)}")

    n = len(golden)
    print("\n" + "=" * 70)
    mode = "full pipeline" if args.full else f"plain retrieve (top_k={args.top_k})"
    print(f"Mode:            {mode}")
    print(f"Questions fully covered: {full_hits}/{n} ({100 * full_hits / n:.0f}%)")
    print(f"Section recall:  {total_found}/{total_expected} ({100 * total_found / total_expected:.0f}%)")
    if misses:
        print("\nMisses:")
        for q, missing, got in misses:
            print(f"  - {q}")
            print(f"    missing §{', §'.join(missing)}; retrieved: {', '.join(got[:10]) or 'nothing'}")


if __name__ == "__main__":
    main()
