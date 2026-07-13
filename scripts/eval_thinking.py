"""
A/B eval: answer generation WITH thinking vs WITHOUT thinking.

For a handful of golden-set questions, run the real query pipeline up to context
assembly ONCE, then generate the answer twice — think=True and think=False — on
the *identical* context so the only variable is the reasoning pass. Captures
latency, token usage (Ollama's own counters), and both full answer texts for
manual quality review, plus an automatic expected-section coverage signal.

Writes eval/thinking_compare_results.json. Meant to be run in the background:
    python scripts/eval_thinking.py
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import TaxCodeRAG  # noqa: E402
import query as Q  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO / "eval" / "thinking_compare_results_real.json"

# A representative spread that stresses where chain-of-thought should matter most
# (calculation / exception / comparison) plus a definitional baseline where it
# likely won't. Matched by exact question text against eval/golden_set.json.
# SELECTED = [
#     "How is the qualified business income deduction calculated?",
#     "How is self-employment tax calculated?",
#     "What are the exceptions to the 10% early withdrawal penalty on retirement accounts?",
#     "How are Roth IRA distributions taxed compared to traditional IRA distributions?",
#     "What counts as a dependent for tax purposes?",
# ]

SELECTED = [
    "Can I have a solo 401k plan as a sole proprietor filing 1040 schedule C?",
    "What are the conditions for vehicles to meet to qualify for a Section 179 deduction in 2025.",
    "What are the limits for equipment deduction for an S-Corporation under Section 179?",
    "What are the % and $ limits for company match for an employee participating in a regular 401k plan?",
    "What are the limits for a Sole Proprietor to deduct business expenses on a Form 1040 Schedule C?",
    "Does a Qualified Business Income deduction apply to state income taxes?",
    "What types of business can take a Qualified Business Deduction and what conditions and limits apply to it?",
    "Can a new busienss deduct expenses in the first year of operations without having any revenues? If so, what limits and conditions apply?",
    "What are the limits to a loss carryover for a business? Does it differ based on the type of business? If so, what are the conditions that apply?",
    "Can a shareholder or partner in a pass through entity deduct estimated taxes as a result of business income?",
    "How much does a sole proprietor pay in payroll taxes for themselves?",
    "as a sole proprietor, do I need to pay state unemployment taxes?",
    "If I'm making $80,000 and I'm married filing separately, how much am i taxed in 2025?",
    "What tax credits can i claim as a student?"
]

_SECTION_IN_TEXT = re.compile(r'(?:§|\bsection\s+)\s*(\d+[A-Za-z]{0,2})', re.IGNORECASE)


def _sections_in(text: str) -> set:
    """IRC section numbers the answer text actually names."""
    return {m.group(1).upper() for m in _SECTION_IN_TEXT.finditer(text)}


def _build_context(rag: TaxCodeRAG, question: str) -> tuple[str, str, list]:
    """Replicate handle_query's retrieval + context assembly (query.py:322-376),
    returning (context, q_type, sources). No answer generation here."""
    q_type = Q._classify_question(rag, question)
    strategy = Q._STRATEGY[q_type]

    cited = Q._extract_cited_sections(question)
    pinned = rag.lookup_sections(cited, max_chunks_per_section=3)[:6]
    for s in pinned:
        s["score"] = 1.0

    raw_expansion = rag.generate(
        strategy["prompt"].format(question=question),
        options={"num_predict": 256}, think=False, label="expand",
    )
    queries = Q._parse_queries(raw_expansion, question)

    seen = {s["id"]: s for s in pinned}
    for q in queries:
        for source in rag.retrieve(q, top_k=strategy["top_k"]):
            sid = source["id"]
            if sid not in seen or source["score"] > seen[sid]["score"]:
                seen[sid] = source
    sources = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:strategy["cap"]]
    sources += rag.expand_refs(sources, max_add=3)

    context = Q._format_context(sources)
    return context, q_type, strategy


def _answer_prompt(strategy: dict, context: str, question: str) -> str:
    """Exactly how handle_query builds the answer prompt (query.py:374-376)."""
    return strategy["answer"].replace(
        "Question: {question}", Q._CONCISE_DIRECTIVE + "Question: {question}"
    ).format(context=context, question=question)


def _generate_answer(rag: TaxCodeRAG, prompt: str, think: bool) -> dict:
    """Call the LLM directly (mirroring generate()'s options) so we can read the
    full response's token/timing counters. num_ctx is pinned to LLM_NUM_CTX just
    like generate(), so no model reload happens between the two modes."""
    opts = {"num_ctx": rag.LLM_NUM_CTX, **Q._GENERATION_OPTIONS}
    t0 = time.time()
    resp = rag._ollama.chat(
        model=rag.ollama_model,
        messages=[{"role": "user", "content": prompt}],
        options=opts, think=think, keep_alive=-1,
    )
    wall = time.time() - t0
    thinking = getattr(resp.message, "thinking", None) or ""
    return {
        "think": think,
        "answer": resp.message.content.strip(),
        "thinking_text": thinking,
        "wall_s": round(wall, 2),
        "load_s": round((getattr(resp, "load_duration", 0) or 0) / 1e9, 2),
        "prefill_tokens": getattr(resp, "prompt_eval_count", None),
        "prefill_s": round((getattr(resp, "prompt_eval_duration", 0) or 0) / 1e9, 2),
        "decode_tokens": getattr(resp, "eval_count", None),
        "decode_s": round((getattr(resp, "eval_duration", 0) or 0) / 1e9, 2),
        "thinking_chars": len(thinking),
    }


def main():
    golden = json.loads((REPO / "eval" / "golden_set.json").read_text())
    # case-insensitive golden-set match; questions not in the golden set still run,
    # just without an expected-sections coverage check
    by_q = {g["question"].strip().lower(): g for g in golden}
    items = [by_q.get(q.strip().lower(), {"question": q, "expected_sections": []})
             for q in SELECTED]
    print(f"Evaluating {len(items)} questions · thinking ON vs OFF\n")

    rag = Q.get_rag()

    # resume from checkpoint: skip questions already evaluated in a previous run
    results = []
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text()).get("results") or []
        if results:
            print(f"Resuming: {len(results)} questions already checkpointed\n")
    done = {r["question"] for r in results}

    for i, g in enumerate(items, 1):
        question = g["question"]
        if question in done:
            continue
        expected = set(s.upper() for s in g["expected_sections"])
        print(f"[{i}/{len(items)}] {question}")

        context, q_type, strategy = _build_context(rag, question)
        prompt = _answer_prompt(strategy, context, question)
        ctx_tokens = len(Q._token_encoder.encode(context))

        # think=True first, then think=False. Prompt is identical, so the 2nd
        # call's prefill may be KV-cached — decode metrics are the clean signal.
        runs = {}
        for think in (True, False):
            r = _generate_answer(rag, prompt, think)
            r["expected_sections_hit"] = sorted(expected & _sections_in(r["answer"]))
            r["expected_sections_missed"] = sorted(expected - _sections_in(r["answer"]))
            runs["think_on" if think else "think_off"] = r
            tag = "ON " if think else "OFF"
            print(f"    think {tag}: {r['wall_s']}s wall · "
                  f"decode {r['decode_tokens']} tok/{r['decode_s']}s · "
                  f"thinking {r['thinking_chars']} chars · "
                  f"hit {r['expected_sections_hit']}")

        results.append({
            "question": question,
            "type": q_type,
            "expected_sections": sorted(expected),
            "context_tokens": ctx_tokens,
            "think_on": runs["think_on"],
            "think_off": runs["think_off"],
        })
        # checkpoint after every question so an interrupted run keeps its data
        RESULTS_PATH.write_text(json.dumps({"summary": None, "results": results}, indent=2))
        print()

    # aggregate latency/token deltas
    def avg(key, mode):
        vals = [r[mode][key] for r in results if r[mode][key] is not None]
        return sum(vals) / len(vals) if vals else 0

    summary = {
        "n": len(results),
        "avg_wall_on": round(avg("wall_s", "think_on"), 1),
        "avg_wall_off": round(avg("wall_s", "think_off"), 1),
        "avg_decode_tokens_on": round(avg("decode_tokens", "think_on")),
        "avg_decode_tokens_off": round(avg("decode_tokens", "think_off")),
        "avg_decode_s_on": round(avg("decode_s", "think_on"), 1),
        "avg_decode_s_off": round(avg("decode_s", "think_off"), 1),
    }
    payload = {"summary": summary, "results": results}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))

    print("=" * 60)
    print(f"avg wall:          ON {summary['avg_wall_on']}s → OFF {summary['avg_wall_off']}s")
    print(f"avg decode tokens: ON {summary['avg_decode_tokens_on']} → OFF {summary['avg_decode_tokens_off']}")
    print(f"avg decode time:   ON {summary['avg_decode_s_on']}s → OFF {summary['avg_decode_s_off']}s")
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
