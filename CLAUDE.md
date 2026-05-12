# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


## Project Overview

Tax Chat is a RAG (Retrieval-Augmented Generation) system that helps **taxpayers and business owners** get answers to questions about the US Tax Code (Title 26 / IRC). It parses `data/usc26.xml`, chunks the XML into token-bounded units that respect legal structure, embeds them into a ChromaDB vector store via Ollama, and answers questions through a Flask web UI.

The query pipeline classifies the incoming question into one of six types (see **Question Types** below), then selects a type-specific retrieval strategy — number of expanded queries, retrieval depth, and answer format — before running hybrid BM25 + semantic retrieval fused via Reciprocal Rank Fusion (RRF), LLM-based reranking, and answer generation.

## Prerequisites

- **Ollama** must be running at `http://localhost:11434` before indexing or serving
- Required models: `gemma` (LLM) and `nomic-custom` (embeddings, built from `Modelfile`)

```bash
ollama pull gemma
ollama pull nomic-embed-text          # base for nomic-custom
ollama create nomic-custom -f Modelfile
```

## Common Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Ingest XML → RAG-ready chunks JSON:**
```bash
python ingest.py                              # defaults: chunk-size=500, overlap=50
python ingest.py --chunk-size 600 --chunk-overlap 30
```

**Build vector index:**
```bash
python rag.py build                           # uses data/rag_chunks2.json by default
python rag.py build data/rag_chunks2.json --index-name my_index --force
python rag.py list                            # list available indexes
```

**Run the web app:**
```bash
python app.py                                 # port 5001, index rag_chunks2
python app.py --port 5000 --index-name rag_chunks2
RAG_INDEX_NAME=rag_chunks2 python app.py
```

**Analysis tools:**
```bash
python analyze_chunks.py data/rag_chunks2.json
python flag_large_chunks.py
```

## Architecture

### Data Flow
```
usc26.xml
  → ingest.py (XMLParser + StructureFirstChunker)
  → data/rag_chunks2.json
  → rag.py build (embed → ChromaDB + BM25 in-memory)
  → app.py / query.py (expand → hybrid retrieve → rerank → generate)
  → Flask web UI (templates/index.html)
```

### Layer Responsibilities

**Ingestion (`ingest.py`, `chunk.py`):**
`ingest.py` orchestrates the full pipeline: it calls `old_scripts/old_ingest.py`'s `XMLParser` to parse the USLM XML into raw chunks, cleans text (strips redundant headings/identifiers, normalizes whitespace), then passes them through `chunk_for_rag_contiguous()` in `chunk.py`. `StructureFirstChunker` targets 300–700 tokens per chunk, splits large sections at subsection boundaries (`(a)`, `(b)`, …), merges small consecutive chunks within the same section, and enforces a hard 1800-token cap. Output is written to `data/rag_chunks2.json`.

**Indexing/Retrieval (`rag.py`):**
`TaxCodeRAG` uses ChromaDB (persistent, in `data/index_<name>/`) for vector storage and `rank_bm25` for in-memory keyword search. On startup it builds or loads the Chroma collection, then constructs a BM25 index over the same documents. `retrieve()` runs both searches in parallel, fetches `top_k * 3` candidates each, and merges them with RRF (`k=60`). Chunk texts are prefixed with `§<identifier> <heading>:` and truncated to 1800 tokens before embedding.

**Query handling (`query.py`):**
`handle_query()` drives the multi-step pipeline:
1. **Classification** — asks the LLM to classify the question into one of six types (see **Question Types**); falls back to keyword heuristics if the LLM response is ambiguous
2. **Strategy selection** — picks a type-specific strategy: expansion prompt, number of queries, retrieval `top_k`, chunk cap, and answer prompt
3. **Query expansion** — asks the LLM to decompose the question into N targeted IRC search queries (N varies by type: 2–5)
4. **Hybrid retrieval** — calls `rag.retrieve(q, top_k=<strategy top_k>)` per expanded query, deduplicates by chunk ID keeping the highest RRF score, caps at `<strategy cap>` chunks
5. **LLM reranking** — asks the LLM to reorder candidates by applicability to the specific question
6. **Answer generation** — formats the top-ranked chunks as context and calls the LLM with the type-specific answer prompt

**Web layer (`app.py`, `templates/index.html`):**
Flask app with two routes: `GET /` renders the chat UI and `POST /api/query` calls `handle_query()`. The RAG instance is initialized at startup (not lazily) so the first request doesn't pay the init cost.

### Key Defaults (in `rag.py`)
| Setting | Value |
|---|---|
| Embedding model | `nomic-custom` |
| LLM | `gemma4:e4b` |
| Chunks file | `data/rag_chunks2.json` |
| Index dir | `data/index_rag_chunks2/` |
| RRF constant k | 60 |
| Max embedding tokens | 1800 |

### Retrieval Parameters by Question Type (in `query.py`)
| Question Type | Queries generated | top_k per query | Max chunks to LLM |
|---|---|---|---|
| application | 3 | 5 | 12 |
| survey | 5 | 7 | 20 |
| exception | 4 | 6 | 18 |
| definitional | 2 | 4 | 10 |
| procedural | 3 | 4 | 10 |
| comparison | 4 | 5 | 16 |

### Chunk JSON Schema (`data/rag_chunks2.json`)
```json
{
  "id": "section-162-a",
  "text": "...",
  "metadata": {
    "identifier": "162",
    "identifiers": ["162", "162(a)"],
    "heading": "Trade or business expenses",
    "tag": "subsection",
    "token_count": 450,
    "parent_id": "section-162",
    "children_ids": ["section-162-a-1"],
    "chunk_index": null,
    "start_char": 0,
    "end_char": 1820
  }
}
```

## Question Types

The classifier in `query.py` assigns every incoming question to one of six types. The type drives everything downstream: how many queries are generated, how deep retrieval goes, and how the answer is structured.

| Type | What the user wants | Example |
|---|---|---|
| **application** | Apply a rule to stated facts — precise calculation, rate, or eligibility result | "What are the % and $ limits for 401k employer match?" / "Can I deduct my home office if I use it 60% for business?" |
| **survey** | Enumerate all options — no specific facts constrain the answer | "What tax credits can I claim as a student?" / "What deductions are available for small businesses?" |
| **exception** | Conditions, limits, phase-outs, or exclusions that constrain a primary rule | "What businesses qualify for the QBI deduction and what limits apply?" / "What are the exceptions to the capital gains exclusion on home sale?" |
| **definitional** | Meaning of a legal term or concept in the IRC | "What counts as a 'qualified business income'?" / "What is the tax definition of a 'dependent'?" |
| **procedural** | How to comply — forms, deadlines, elections, penalties for non-compliance | "How do I elect S-corp status?" / "When is the deadline to fund a SEP-IRA?" |
| **comparison** | Side-by-side analysis of two or more provisions, entities, or strategies | "What is the tax difference between an LLC and an S-corp?" / "Traditional IRA vs. Roth IRA — how do they compare?" |

**Classification fallback:** if the LLM response is ambiguous, `_classify_question()` falls back to keyword heuristics. Questions containing "what credits / what deductions / what options / what can I claim" → survey; "conditions / limits / exceptions / phase-out / excluded" → exception; "what is / what counts as / definition of" → definitional; "how do I / when do I / what form / deadline" → procedural; "vs / versus / difference between / compare" → comparison; everything else → application.

## Environment Variables

| Variable | Purpose |
|---|---|
| `RAG_INDEX_NAME` | Override index name (maps to `data/index_<name>/`) |
| `RAG_CHUNKS_FILE` | Override chunks file path |
