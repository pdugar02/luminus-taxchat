#!/usr/bin/env python3
"""
Evaluate a RAG answer and get prompt improvement suggestions via Claude.

Usage:
    python eval_improve.py "What education tax credits are available?"
    python eval_improve.py "What is the capital gains tax rate?" --json
"""

import argparse
import json
import os
import re
import sys

import anthropic
import dotenv

from query import RERANK_PROMPT, _STRATEGY, handle_query

dotenv.load_dotenv()

EVAL_SYSTEM = (
    "You are an expert in US tax law and AI system evaluation. "
    "You evaluate answers produced by a RAG system that queries the US Internal Revenue Code (Title 26)."
)

EVAL_USER_PROMPT = """Evaluate this RAG answer for a tax question.

## Question
{question}

## Question type classified by the system: {q_type}

## Answer produced by RAG system
{answer}

## Source chunks retrieved (top 5)
{sources}

## Current system prompts used for this question type (for context when suggesting edits)

### {expansion_name}
{expansion_prompt}

### RERANK_PROMPT
{rerank_prompt}

### {answer_name}
{answer_prompt}

---

Score the answer on each dimension from 1 to 5:
- citation_accuracy: Are IRC section numbers cited correctly and present in the sources?
- completeness: Does it cover all relevant aspects of the question?
- step_by_step_clarity: For calculation questions, are steps clearly shown?
- legal_precision: Is the legal language accurate and appropriately qualified?
- appropriate_hedging: Does it recommend professional consultation where warranted?

For any dimension scoring below 4, provide specific, actionable edits to the relevant prompt(s).
Use the exact prompt variable names shown above (e.g. {expansion_name}, {answer_name}) as keys.

Return ONLY valid JSON — no markdown fences, no explanation:
{{
  "scores": {{
    "citation_accuracy": <int 1-5>,
    "completeness": <int 1-5>,
    "step_by_step_clarity": <int 1-5>,
    "legal_precision": <int 1-5>,
    "appropriate_hedging": <int 1-5>
  }},
  "overall": <int 1-5>,
  "critique": "<2-3 sentence overall assessment>",
  "prompt_suggestions": {{
    "{expansion_name}": "<specific instruction additions/changes, or null>",
    "RERANK_PROMPT": "<specific instruction additions/changes, or null>",
    "{answer_name}": "<specific instruction additions/changes, or null>"
  }}
}}"""


def evaluate(question: str, answer: str, sources: list[dict], q_type: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    strategy = _STRATEGY[q_type]
    expansion_name = f"EXPANSION_PROMPT_{q_type.upper()}"
    answer_name = f"ANSWER_PROMPT_{q_type.upper()}"

    sources_text = "\n\n".join(
        f"[{i + 1}] §{s.get('metadata', {}).get('identifier', '?')}: {s.get('text', '')[:300]}..."
        for i, s in enumerate(sources[:5])
    )

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": EVAL_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": EVAL_USER_PROMPT.format(
                    question=question,
                    answer=answer,
                    sources=sources_text,
                    q_type=q_type,
                    expansion_name=expansion_name,
                    expansion_prompt=strategy["prompt"],
                    rerank_prompt=RERANK_PROMPT,
                    answer_name=answer_name,
                    answer_prompt=strategy["answer"],
                ),
            }
        ],
    )

    raw = message.content[0].text
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    return json.loads(cleaned)


def print_report(question: str, result: dict) -> None:
    scores = result.get("scores", {})
    overall = result.get("overall", "?")
    critique = result.get("critique", "")
    suggestions = result.get("prompt_suggestions", {})

    print(f"\n{'=' * 60}")
    print("EVALUATION REPORT")
    print(f"{'=' * 60}")
    print(f"Question: {question}\n")
    print(f"Overall score: {overall}/5")
    print("\nScores:")
    for k, v in scores.items():
        bar = "█" * v + "░" * (5 - v)
        flag = " ⚠" if v < 4 else ""
        print(f"  {k:<28} {bar} {v}/5{flag}")
    print(f"\nCritique:\n  {critique}")

    has_suggestions = any(v for v in suggestions.values() if v)
    if has_suggestions:
        print(f"\n{'─' * 60}")
        print("PROMPT IMPROVEMENT SUGGESTIONS")
        print(f"{'─' * 60}")
        for prompt_name, suggestion in suggestions.items():
            if suggestion:
                print(f"\n[{prompt_name}]")
                for line in suggestion.splitlines():
                    print(f"  {line}")
    else:
        print("\n✓ No prompt changes suggested.")

    print(f"\n{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a RAG answer and suggest prompt improvements via Claude"
    )
    parser.add_argument("question", help="Tax question to evaluate")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted report")
    args = parser.parse_args()

    print(f"Running query: {args.question!r}")
    payload, status = handle_query({"question": args.question})
    if status != 200:
        print(f"Query failed: {payload.get('error')}", file=sys.stderr)
        sys.exit(1)

    answer = payload["answer"]
    sources = payload["sources"]
    q_type = payload["question_type"]

    print(f"Question type: {q_type}")
    print("Evaluating with Claude...")
    result = evaluate(args.question, answer, sources, q_type)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(args.question, result)


if __name__ == "__main__":
    main()
