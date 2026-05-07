"""
RAG query logic for the Tax Code web app.
Provides get_rag and request handlers used by app.py.
"""

import json
import re
import time
from pathlib import Path

from rag import TaxCodeRAG

rag = None

EXPANSION_PROMPT = """You are an expert in US tax law (Title 26, Internal Revenue Code). Break down this tax question into 3 targeted search queries that together will surface ALL relevant code sections, including the current version of any amended provisions.

Question: {question}

Think through:
- What is the core tax issue? (e.g., deductibility, income recognition, gain/loss, credit eligibility, entity treatment, procedural requirement)
- Which IRC sections directly govern this? Consider the primary rule AND sections that amend, limit, or supersede it. Broad areas for reference: §§1-59 (income taxes), §§61-140 (gross income/exclusions), §§161-291 (deductions/disallowances), §§301-385 (corporate), §§401-436 (retirement), §§501-530 (exempt orgs), §§701-761 (partnerships), §§901-908 (foreign tax credits), §§1001-1298 (gains/losses/capital gains), §§1361-1379 (S corps), §§6001-7000 (procedure).
- Has this area been amended by TCJA 2017, CARES 2020, IRA 2022, or another act? If so, phrase query 2 to match the TEXT of that amended provision — what words would appear in that specific subsection?
- What definitions, thresholds, or eligibility tests does the answer turn on? What exceptions or special rules apply to the specific entity type or transaction?

Write each query as a short natural language phrase using the legal terminology that would appear in the relevant IRC section itself. Do NOT use boolean operators (AND, OR), quotation marks around terms, or search engine syntax.

Query 1: The primary statutory rule or computation governing the issue
Query 2: The amended or superseding provision — phrased as the actual content of that section (e.g., "tax rate tables for taxable years beginning after December 31 2017 married filing separately"), not as a general search for amendments
Query 3: Key definitions, thresholds, exceptions, or cross-referenced rules the answer depends on

Return ONLY a valid JSON array of exactly 3 plain strings. No markdown, no code fences, no explanation — just the array:
["query 1", "query 2", "query 3"]
"""

RERANK_PROMPT = """You are a US tax law expert. Given a tax question and a list of retrieved code sections, reorder them from most to least directly applicable to the specific facts — especially the year, filing status, entity type, and transaction type in the question.

Question: {question}

Retrieved sections (ID: first line of text):
{summaries}

Return a JSON array of ALL IDs in order from most to least relevant. Include every ID exactly once.
Return ONLY the JSON array, no explanation:
["id1", "id2", ...]
"""

ANSWER_PROMPT = """You are an expert tax law assistant helping IRS employees apply the US Tax Code (Title 26).

Critical rules:
- When sections conflict or one amends another, ALWAYS prefer the provision with the later effective date. For example, a section that applies "for taxable years beginning after December 31, 2017" supersedes the original table for tax year 2018 and later.
- Show step-by-step calculations where applicable.
- Cite specific section numbers (e.g., §1(j), §63(c)(2)) for every rule you apply.
- If a retrieved section was superseded or only applies to specific years, state this explicitly and use the correct current provision instead.
- If you cannot find a complete answer from the provided sections, explain exactly what information is missing.

Tax Code Sections:
{context}

Question: {question}

Answer (step by step where calculations are involved):"""


def _parse_queries(raw: str, fallback: str) -> list[str]:
    """Parse LLM output into a list of query strings, falling back to raw text."""
    # strip markdown code fences the LLM sometimes wraps output in
    cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', raw).strip()
    # greedy match to capture the full array (non-greedy would stop at the first ])
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            queries = json.loads(match.group())
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return [q for q in queries[:5] if q.strip()]
        except json.JSONDecodeError:
            pass
    # fallback: extract any double-quoted strings long enough to be a real query
    quoted = re.findall(r'"([^"]{15,})"', cleaned)
    if quoted:
        return [q for q in quoted[:5] if q.strip()]
    return [cleaned or fallback]


def _rerank_sources(rag_system: TaxCodeRAG, question: str, sources: list[dict]) -> list[dict]:
    """Ask the LLM to reorder sources by applicability to the question."""
    summaries = "\n".join(
        f"{s['id']}: {s['text'][:150].replace(chr(10), ' ')}"
        for s in sources
    )
    raw = rag_system.generate(RERANK_PROMPT.format(question=question, summaries=summaries))
    cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', raw).strip()
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            reranked_ids = json.loads(match.group())
            if isinstance(reranked_ids, list):
                id_to_source = {s['id']: s for s in sources}
                reranked = [id_to_source[rid] for rid in reranked_ids if rid in id_to_source]
                # append any sources the reranker dropped (safety net)
                mentioned = {rid for rid in reranked_ids if rid in id_to_source}
                reranked += [s for s in sources if s['id'] not in mentioned]
                print(f"Reranked: {[s['id'] for s in reranked[:5]]}")
                return reranked
        except (json.JSONDecodeError, KeyError):
            pass
    print("Reranking failed, using original order")
    return sources


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
    if not question:
        return {"error": "Question is required", "sources": []}, 400

    rag_system = get_rag()
    start = time.time()

    # generate multiple targeted queries covering different aspects of the question
    raw_expansion = rag_system.generate(EXPANSION_PROMPT.format(question=question))
    queries = _parse_queries(raw_expansion, question)
    print(f"Expanded into {len(queries)} queries: {queries}")

    # retrieve for each query, deduplicate by chunk ID keeping the highest score
    seen: dict[str, dict] = {}
    for q in queries:
        for source in rag_system.retrieve(q, top_k=5):
            sid = source["id"]
            if sid not in seen or source["score"] > seen[sid]["score"]:
                seen[sid] = source

    # sort by score descending, cap at 15 chunks for context window
    sources = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:15]

    # rerank by applicability to the specific question before passing to the answer LLM
    sources = _rerank_sources(rag_system, question, sources)

    context = "\n\n".join(
        f"[§{s['metadata'].get('identifier', '?')}] {s['text']}"
        for s in sources
    )
    answer = rag_system.generate(ANSWER_PROMPT.format(context=context, question=question))

    print(f"Query took {time.time() - start:.1f}s")
    return {
        "answer": answer,
        "sources": [rag_system.format_source(s) for s in sources],
    }, 200
