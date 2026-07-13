"""
RAG query logic for the Tax Code web app.
Provides get_rag and request handlers used by app.py.
"""

import json
import re
import time
from pathlib import Path

import tiktoken

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


# ── Follow-up condensation ────────────────────────────────────────────────────

CONDENSE_PROMPT = """Given the conversation below, rewrite the user's latest message as a single, fully self-contained tax question. Resolve pronouns and implicit references ("it", "that deduction", "what about for X") using the conversation. Keep every specific fact that remains relevant (amounts, filing status, entity type, IRC sections mentioned). Return ONLY the rewritten question — no explanation.

Conversation:
{history}

Latest message: {question}

Standalone question:"""



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

# The model writes citations and formulas in LaTeX ($\S 1401(\text{a})$,
# $\frac{1}{2}$), which the UI's markdown renderer passes through as literal
# text. Convert to plain Unicode before returning the answer. Real currency
# amounts ($250,000) are left untouched.
_LATEX_SYMBOLS = [
    (re.compile(r"\\frac\{1\}\{2\}"), "½"),
    (re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}"), r"\1/\2"),
    (re.compile(r"\\S(?![A-Za-z])"), "§"),
    (re.compile(r"\\times\b"), "×"),
    (re.compile(r"\\geq?\b"), "≥"),
    (re.compile(r"\\leq?\b"), "≤"),
    (re.compile(r"\\(?:rightarrow|to)\b|\\xrightarrow\{[^{}]*\}"), "→"),
    (re.compile(r"\\Rightarrow\b"), "⇒"),
    (re.compile(r"\\cdot\b"), "·"),
    (re.compile(r"\\approx\b"), "≈"),
    (re.compile(r"\\pm\b"), "±"),
    (re.compile(r"\\%"), "%"),
    (re.compile(r"\\quad\b|\\,|\\;"), " "),
    (re.compile(r"\\(?:left|right)(?=[(){}\[\]|.])"), ""),
]


def _clean_latex(text: str) -> str:
    """Convert the model's LaTeX-ish notation to plain Unicode for display."""
    # A math span wrapping an escaped currency amount ($\$2,500,000$) is the model
    # quoting a dollar figure "mathematically" — unwrap it to a plain $2,500,000 now,
    # before the \$-protection below strips the backslash this shape is recognized by.
    text = re.sub(r"\$\\\$([^$\n]*?)\$", r"$\1", text)
    text = text.replace("\\$", "\x00")  # protect remaining escaped (real) dollar signs
    # unwrap math delimiters: $$...$$ blocks, then $...$ spans that are clearly math —
    # they contain a LaTeX command, a comparator, or a bare section citation. A lone
    # $250,000 or a "$5,000 to $10,000" range never matches these shapes.
    text = re.sub(r"\$\$([^$]+)\$\$", r"\1", text)
    text = re.sub(r"\$([^$\n]*\\[^$\n]*)\$", r"\1", text)
    text = re.sub(r"\$([^$\n]*[<>=≤≥][^$\n]*)\$", r"\1", text)
    text = re.sub(r"\$(\d+(?:\([^)$\n]*\))+)\$", r"\1", text)
    for _ in range(2):  # unwrap nested \text{\text{...}}
        text = re.sub(r"\\(?:text|mathrm|mathbf|mathit|mathsf)\{([^{}]*)\}", r"\1", text)
    for pattern, repl in _LATEX_SYMBOLS:
        text = pattern.sub(repl, text)
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)  # drop any leftover \commands
    text = re.sub(r"§\s+(?=\d)", "§", text)  # "§ 162" → "§162", matching app style
    return text.replace("\x00", "$")


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


# Matches "§162", "section 199A", "sec. 401" in a user question.
_CITED_SECTION_RE = re.compile(r'(?:§|\bsec(?:tion)?s?\.?\s+)\s*(\d+[A-Za-z]{0,2}(?:-\d+)?)', re.IGNORECASE)


def _extract_cited_sections(question: str) -> list[str]:
    """Return IRC section numbers the user explicitly cited, in order, deduplicated."""
    return list(dict.fromkeys(m.group(1).upper() if m.group(1)[-1].isalpha() else m.group(1)
                              for m in _CITED_SECTION_RE.finditer(question)))


def _format_history(history: list[dict], max_messages: int = 6, max_chars: int = 500) -> str:
    """Render recent conversation turns as plain text for prompting."""
    lines = []
    for msg in history[-max_messages:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = (msg.get("content") or "").strip()
        if len(content) > max_chars:
            content = content[:max_chars] + " […]"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ── Follow-up detection (deterministic condensation gate) ─────────────────────
# Condensation is a full LLM call that reads the whole transcript, so it costs
# 30–50s and grows every turn. We only pay it when the question actually depends
# on prior turns — i.e. it carries an unresolved reference: a leading connective
# / ellipsis ("what about…", "and if…"), an anaphoric pronoun/demonstrative
# ("it", "that", "those"), or a bare subjectless fragment. A self-contained
# question (even an unrelated topic change) has nothing for condensation to
# resolve and is passed through unchanged. The gate deliberately errs toward
# condensing when ambiguous: a needless rewrite only wastes time, whereas a
# missed follow-up would retrieve on an unresolvable question.

_FOLLOWUP_LEAD_RE = re.compile(
    r"^\s*(what about|how about|what if|what else|anything else|"
    r"and\b|but\b|also\b|then\b|or\b|so\b|"
    r"for (those|these|them|it|that|this)\b)",
    re.IGNORECASE,
)

# Anaphoric pronouns/demonstratives that point back to an earlier turn.
_ANAPHORA_RE = re.compile(
    r"\b(it|its|it's|that|this|these|those|them|they|their|theirs|such|"
    r"the same|the former|the latter|the above|the previous|the prior)\b",
    re.IGNORECASE,
)

# Leading words that mark a self-contained interrogative (so a short question
# like "What is AMT?" isn't mistaken for an elliptical fragment).
_QUESTION_WORD_RE = re.compile(
    r"^\s*(what|how|when|where|why|which|who|whom|whose|can|could|do|does|did|"
    r"is|are|was|were|should|would|will|may|must|am|if)\b",
    re.IGNORECASE,
)


def _needs_condensation(question: str) -> bool:
    """Deterministically decide whether a question depends on conversation history.
    Returns True only for likely follow-ups; self-contained questions skip the
    expensive condensation LLM call. No LLM/embedding calls — pure string work."""
    q = question.strip()
    if _FOLLOWUP_LEAD_RE.search(q):
        return True
    if _ANAPHORA_RE.search(q):
        return True
    # very short and not phrased as a standalone question → likely an ellipsis
    # ("contribution limits?", "for married couples?")
    if len(q.split()) <= 4 and not _QUESTION_WORD_RE.search(q):
        return True
    return False


def _condense_question(rag_system: TaxCodeRAG, question: str, history: list[dict]) -> str:
    """Rewrite a follow-up into a standalone question using conversation history."""
    raw = rag_system.generate(
        CONDENSE_PROMPT.format(history=_format_history(history), question=question),
        options={"temperature": 0},
    )
    standalone = raw.strip().strip('"')
    # guard against a degenerate rewrite (empty or runaway output)
    if not standalone or len(standalone) > 4 * max(len(question), 100):
        return question
    return standalone


def _classify_question(rag_system: TaxCodeRAG, question: str) -> str:
    """Ask the LLM to classify the question type; fall back to keyword heuristics."""
    raw = rag_system.generate(
        CLASSIFY_PROMPT.format(question=question),
        options={"num_predict": 10, "temperature": 0},
        label="classify",
        # think=False
    )
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


# Keep the statute context within the model's window: chunks can be up to 1800
# tokens each, and overflowing num_ctx makes Ollama silently truncate the prompt
# (observed as empty answers). Sources are ranked, so the lowest-ranked chunks
# are dropped first.
_CONTEXT_TOKEN_BUDGET = 5000
_GENERATION_OPTIONS = {"num_ctx": 16384}

# Appended to the answer prompt to trim the written half of generation (the model
# was decoding ~1000 content tokens per answer). Thinking stays on for accuracy.
_CONCISE_DIRECTIVE = (
    "Keep your answer concise and focused: lead with the direct answer, include only "
    "the rules, figures, and section numbers that matter, and do not restate the question "
    "or quote statute text at length.\n\n"
)

_token_encoder = tiktoken.get_encoding("cl100k_base")


def _format_context(sources: list[dict]) -> str:
    """Render sources as the Tax Code Sections block, including where each section
    lives in the code and its effective-date note when available. Stops adding
    chunks once the token budget is reached."""
    blocks = []
    used_tokens = 0
    for s in sources:
        meta = s.get("metadata", {})
        header = f"[§{meta.get('identifier', '?')}]"
        if meta.get("hierarchy_path"):
            header += f" ({meta['hierarchy_path']})"
        block = f"{header} {s['text']}"
        note = (meta.get("effective_date_note") or "").strip()
        if note:
            block += f"\n(Effective date note: {note[:300]})"
        block_tokens = len(_token_encoder.encode(block))
        if blocks and used_tokens + block_tokens > _CONTEXT_TOKEN_BUDGET:
            break
        used_tokens += block_tokens
        blocks.append(block)
    return "\n\n".join(blocks)


def handle_query(data: dict) -> tuple[dict, int]:
    """Handle query requests and return (payload, status_code).

    Accepts an optional `history` list of {"role": "user"|"assistant", "content": str}
    messages; follow-up questions are condensed into standalone questions before
    retrieval so the pipeline works mid-conversation.
    """
    question = data.get("question", "").strip()
    history = data.get("history") or []
    retrieve_only = data.get("retrieve_only", False)
    if not question:
        return {"error": "Question is required", "sources": []}, 400

    rag_system = get_rag()
    start = time.time()

    # resolve follow-ups against conversation history — but only when the question
    # actually references prior turns; condensation is an expensive LLM call
    if history and _needs_condensation(question):
        t0 = time.time()
        question = _condense_question(rag_system, question, history)
        print(f"Condense:  {time.time() - t0:.1f}s — {question}")
    elif history:
        print("Condense:  skipped (self-contained question)")

    # classify and select type-specific strategy
    t0 = time.time()
    q_type = _classify_question(rag_system, question)
    strategy = _STRATEGY[q_type]
    print(f"Classify:  {time.time() - t0:.1f}s — {q_type}")

    # exact-citation path: sections the user explicitly named are always included
    cited = _extract_cited_sections(question)
    pinned = rag_system.lookup_sections(cited, max_chunks_per_section=3)[:6]
    for s in pinned:
        s["score"] = 1.0  # pinned above any RRF score
    if pinned:
        print(f"Pinned:    {len(pinned)} chunks for cited §{', §'.join(cited)}")

    # expand the question into targeted IRC search queries — no chain-of-thought
    # needed here, so skip the model's (expensive) thinking and cap the output
    t0 = time.time()
    raw_expansion = rag_system.generate(
        strategy["prompt"].format(question=question),
        options={"num_predict": 256}, think=False, label="expand",
    )
    queries = _parse_queries(raw_expansion, question)
    print(f"Expansion: {time.time() - t0:.1f}s — {len(queries)} queries:")
    for q in queries:
        print(f"  {q}")

    # retrieve for each query; deduplicate by chunk ID keeping the highest RRF score
    t0 = time.time()
    seen: dict[str, dict] = {s["id"]: s for s in pinned}
    for q in queries:
        for source in rag_system.retrieve(q, top_k=strategy["top_k"]):
            sid = source["id"]
            if sid not in seen or source["score"] > seen[sid]["score"]:
                seen[sid] = source
    sources = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:strategy["cap"]]

    # one-hop cross-reference expansion: pull in sections the top chunks cite
    sources += rag_system.expand_refs(sources, max_add=3)
    print(f"Retrieval: {time.time() - t0:.1f}s — {len(sources)} chunks")

    formatted_sources = [rag_system.format_source(s) for s in sources]

    if retrieve_only:
        return {
            "answer": f"Retrieved {len(sources)} chunks ({q_type} strategy, retrieve_only mode).",
            "sources": formatted_sources,
            "question_type": q_type,
        }, 200

    context = _format_context(sources)
    # inject the concision directive just before the "Question:" cue in every answer template
    answer_prompt = strategy["answer"].replace(
        "Question: {question}", _CONCISE_DIRECTIVE + "Question: {question}"
    ).format(context=context, question=question)
    t0 = time.time()
    answer = rag_system.generate(answer_prompt, options=_GENERATION_OPTIONS, think=False, label="answer")
    print(f"Answer:    {time.time() - t0:.1f}s")

    # Verify/retry pass disabled for latency: it costs ~56s (mostly the model's thinking)
    # and, being fail-open, mostly rubber-stamps PASS. Commented out rather than removed so
    # it can be restored if answer quality regresses.
    # t0 = time.time()
    # verdict = rag_system.generate(
    #     VERIFY_PROMPT.format(question=question, answer=answer, context=context),
    #     options=_GENERATION_OPTIONS,
    #     label="verify",
    # )
    # feedback = verdict.strip()
    # print(f"Verify:    {time.time() - t0:.1f}s — {feedback[:80]}")
    # # only retry on substantive feedback; an empty verdict is a verifier failure, not a rejection
    # if feedback and feedback.upper() != "PASS":
    #     t0 = time.time()
    #     # keep the feedback out of the statute context so it can't be cited as source text
    #     retry = rag_system.generate(strategy["answer"].format(
    #         context=context,
    #         question=f"{question}\n\n(A previous draft of this answer was rejected because: {feedback} — make sure this answer fixes that.)",
    #     ), options=_GENERATION_OPTIONS, label="retry")
    #     if retry.strip():
    #         answer = retry
    #     print(f"Retry:     {time.time() - t0:.1f}s")

    print(f"Total:     {time.time() - start:.1f}s")
    return {
        "answer": _clean_latex(answer),
        "sources": formatted_sources,
        "question_type": q_type,
        "standalone_question": question,
    }, 200
