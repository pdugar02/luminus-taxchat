"""
RAG query logic for the Tax Code web app.
Provides get_rag and request handlers used by app.py.
"""

import json
import re
import time
from pathlib import Path

from rag import TaxCodeRAG
from expansion_prompts import (
    EXPANSION_PROMPT_APPLICATION, EXPANSION_PROMPT_SURVEY, EXPANSION_PROMPT_EXCEPTION,
    EXPANSION_PROMPT_DEFINITIONAL, EXPANSION_PROMPT_PROCEDURAL, EXPANSION_PROMPT_COMPARISON,
)
from answer_prompts import (
    ANSWER_PROMPT_APPLICATION, ANSWER_PROMPT_SURVEY, ANSWER_PROMPT_EXCEPTION,
    ANSWER_PROMPT_DEFINITIONAL, ANSWER_PROMPT_PROCEDURAL, ANSWER_PROMPT_COMPARISON,
)

rag = None

# ── Question type constants ───────────────────────────────────────────────────

Q_APPLICATION  = "application"   # apply a rule to stated facts → precise result
Q_SURVEY       = "survey"        # enumerate all options (no constraining facts)
Q_EXCEPTION    = "exception"     # conditions, limits, phase-outs, exclusions
Q_DEFINITIONAL = "definitional"  # meaning of a legal term or concept
Q_PROCEDURAL   = "procedural"    # how to comply — forms, deadlines, elections
Q_COMPARISON   = "comparison"    # side-by-side analysis of two provisions/strategies

_ALL_TYPES = {Q_APPLICATION, Q_SURVEY, Q_EXCEPTION, Q_DEFINITIONAL, Q_PROCEDURAL, Q_COMPARISON}


# ── Classification ────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """Classify this tax question into exactly one category. Return ONLY the single category word — nothing else.

application  — The user states specific facts and wants a precise result: a dollar amount, percentage, rate, or yes/no eligibility determination. The goal is applying a rule to their situation.
               Example: "What are the % and $ limits for 401k employer match?" / "Can I deduct my car if I use it 70% for business?"

survey       — Open-ended: the user wants to know what options, credits, deductions, or strategies exist. No specific facts constrain the answer — the goal is enumerating possibilities.
               Example: "What tax credits can I claim as a student?" / "What deductions are available for small businesses?"

exception    — The user asks about conditions, limits, phase-outs, or exclusions that modify or constrain a primary rule. The goal is understanding what restricts or qualifies a benefit.
               Example: "What businesses qualify for the QBI deduction and what limits apply?" / "What are the exceptions to the home sale exclusion?"

definitional — The user wants to know what a legal term or concept means under the IRC. The goal is understanding the definition itself, not applying it.
               Example: "What counts as 'qualified business income'?" / "What is the tax definition of a 'dependent'?"

procedural   — The user wants to know how to comply: what steps to take, what form to file, what deadline applies, or what happens if they don't comply. The goal is administrative action, not a tax calculation.
               Example: "How do I elect S-corp status?" / "When is the deadline to fund a SEP-IRA?" / "What form do I use to report foreign income?"

comparison   — The user wants a side-by-side analysis of two or more provisions, entity types, account types, or strategies.
               Example: "Traditional IRA vs. Roth IRA — how do they compare?" / "What is the tax difference between an LLC and an S-corp?"

Disambiguation note: "How do I calculate X?" is application (the goal is the number). "How do I file/elect/report X?" is procedural (the goal is the administrative action).

Question: {question}
Category:"""


# ── Verification ─────────────────────────────────────────────────────────────

VERIFY_PROMPT = """Question: {question}
Answer: {answer}
Sources: {context}

Does the answer address the question and is it supported by the sources?
If yes: output exactly: PASS
If no: output one sentence describing the specific problem.
Output nothing else."""



# ── Strategy routing table ────────────────────────────────────────────────────

_STRATEGY = {
    Q_APPLICATION:  {"prompt": EXPANSION_PROMPT_APPLICATION,  "top_k": 5, "cap": 6,  "answer": ANSWER_PROMPT_APPLICATION},
    Q_SURVEY:       {"prompt": EXPANSION_PROMPT_SURVEY,       "top_k": 7, "cap": 8,  "answer": ANSWER_PROMPT_SURVEY},
    Q_EXCEPTION:    {"prompt": EXPANSION_PROMPT_EXCEPTION,    "top_k": 6, "cap": 8,  "answer": ANSWER_PROMPT_EXCEPTION},
    Q_DEFINITIONAL: {"prompt": EXPANSION_PROMPT_DEFINITIONAL, "top_k": 4, "cap": 5,  "answer": ANSWER_PROMPT_DEFINITIONAL},
    Q_PROCEDURAL:   {"prompt": EXPANSION_PROMPT_PROCEDURAL,   "top_k": 4, "cap": 5,  "answer": ANSWER_PROMPT_PROCEDURAL},
    Q_COMPARISON:   {"prompt": EXPANSION_PROMPT_COMPARISON,   "top_k": 5, "cap": 8,  "answer": ANSWER_PROMPT_COMPARISON},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_queries(raw: str, fallback: str) -> list[str]:
    """Parse LLM output into a list of query strings, falling back to raw text."""
    cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', raw).strip()
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            queries = json.loads(match.group())
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return [q for q in queries[:6] if q.strip()]
        except json.JSONDecodeError:
            pass
    quoted = re.findall(r'"([^"]{15,})"', cleaned)
    if quoted:
        return [q for q in quoted[:6] if q.strip()]
    return [cleaned or fallback]


def _classify_question(rag_system: TaxCodeRAG, question: str) -> str:
    """Ask the LLM to classify the question type; fall back to keyword heuristics."""
    raw = rag_system.generate(CLASSIFY_PROMPT.format(question=question), options={"num_predict": 10, "temperature": 0})
    words = raw.strip().split()
    first_word = words[0].lower().rstrip(".,:") if words else ""
    if first_word in _ALL_TYPES:
        return first_word

    lower = question.lower()
    if any(kw in lower for kw in ["vs ", " vs.", "versus", "difference between", "compare "]):
        return Q_COMPARISON
    if any(kw in lower for kw in ["what credits", "what deductions", "what options", "what can i claim", "what are all", "what types of", "what are the ways", "what benefits"]):
        return Q_SURVEY
    if any(kw in lower for kw in ["what counts as", "definition of", "what is a ", "what qualifies as", "define ", "what does it mean"]):
        return Q_DEFINITIONAL
    if any(kw in lower for kw in ["how do i ", "how to ", "when do i ", "what form", "deadline", " elect ", "penalty for", "how to file", "how to report"]):
        return Q_PROCEDURAL
    if any(kw in lower for kw in ["conditions", " limits", "exceptions", "excluded from", "phase-out", "what conditions", "what limits", "restrictions", "qualify for"]):
        return Q_EXCEPTION
    return Q_APPLICATION



# ── Public API ────────────────────────────────────────────────────────────────

def get_rag(index_name: str = None, chunks_file: str = None) -> TaxCodeRAG:
    """Lazy initialization of the RAG system."""
    global rag
    if rag is None:
        data_dir = Path(__file__).parent / "data"

        if index_name:
            index_dir = data_dir / f"index_{index_name}"
        elif chunks_file:
            index_dir = data_dir / f"index_{Path(chunks_file).stem}"
        else:
            index_dir = data_dir / "index_rag_chunks2"
            chunks_file = str(data_dir / "rag_chunks2.json")

        rag = TaxCodeRAG(
            chunks_path=chunks_file,
            index_dir=str(index_dir),
        )
    return rag


def handle_query(data: dict) -> tuple[dict, int]:
    """Handle query requests and return (payload, status_code)."""
    question = data.get("question", "").strip()
    retrieve_only = data.get("retrieve_only", False)
    if not question:
        return {"error": "Question is required", "sources": []}, 400

    rag_system = get_rag()
    start = time.time()

    # classify and select type-specific strategy
    t0 = time.time()
    q_type = _classify_question(rag_system, question)
    strategy = _STRATEGY[q_type]
    print(f"Classify:  {time.time() - t0:.1f}s — {q_type}")

    # expand the question into targeted IRC search queries
    t0 = time.time()
    raw_expansion = rag_system.generate(strategy["prompt"].format(question=question))
    queries = _parse_queries(raw_expansion, question)
    print(f"Expansion: {time.time() - t0:.1f}s — {len(queries)} queries:")
    for q in queries:
        print(f"  {q}")

    # retrieve for each query; deduplicate by chunk ID keeping the highest RRF score
    t0 = time.time()
    seen: dict[str, dict] = {}
    for q in queries:
        for source in rag_system.retrieve(q, top_k=strategy["top_k"]):
            sid = source["id"]
            if sid not in seen or source["score"] > seen[sid]["score"]:
                seen[sid] = source
    sources = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:strategy["cap"]]
    print(f"Retrieval: {time.time() - t0:.1f}s — {len(sources)} chunks")

    formatted_sources = [rag_system.format_source(s) for s in sources]

    if retrieve_only:
        return {
            "answer": f"Retrieved {len(sources)} chunks ({q_type} strategy, retrieve_only mode).",
            "sources": formatted_sources,
            "question_type": q_type,
        }, 200

    context = "\n\n".join(
        f"[§{s['metadata'].get('identifier', '?')}] {s['text']}"
        for s in sources
    )
    # print(context)
    t0 = time.time()
    answer = rag_system.generate(strategy["answer"].format(context=context, question=question))
    print(f"Answer:    {time.time() - t0:.1f}s")

    t0 = time.time()
    verdict = rag_system.generate(VERIFY_PROMPT.format(question=question, answer=answer, context=context))
    print(f"Verify:    {time.time() - t0:.1f}s — {verdict.strip()[:80]}")
    if verdict.strip().upper() != "PASS":
        feedback = verdict.strip()
        t0 = time.time()
        answer = rag_system.generate(strategy["answer"].format(
            context=f"Note: a previous attempt had this issue: {feedback}. Fix it.\n\n{context}",
            question=question,
        ))
        print(f"Retry:     {time.time() - t0:.1f}s")

    print(f"Total:     {time.time() - start:.1f}s")
    return {
        "answer": answer,
        "sources": formatted_sources,
        "question_type": q_type,
    }, 200
