# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tax Chat is a RAG (Retrieval-Augmented Generation) system for IRS employees to query the US Tax Code (Title 26). It parses `data/usc26.xml`, chunks it into semantically coherent units, embeds them into a ChromaDB vector store via Ollama, and serves answers through a Flask web UI.

## Prerequisites

- **Ollama** must be running at `http://localhost:11434` before any indexing or serving
- Required models: `llama3.1:8b` (LLM) and `nomic-custom` (embeddings, built from `Modelfile`)

```bash
ollama pull llama3.1:8b
ollama create nomic-custom -f Modelfile
```

## Common Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Ingest XML → chunks JSON:**
```bash
python ingest.py                              # default settings
python ingest.py --chunk-size 600 --chunk-overlap 30
```

**Build vector index:**
```bash
python rag.py build                           # uses data/rag_chunks2.json by default
python rag.py build data/rag_chunks.json --index-name my_index --force
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
usc26.xml → ingest.py → chunks.json → chunk.py → rag_chunks2.json → rag.py → ChromaDB → app.py → web UI
```

### Layer Responsibilities

**Ingestion (`ingest.py`, `chunk.py`):** Parses USLM XML preserving legal hierarchy (sections → subsections → paragraphs). `StructureFirstChunker` in `chunk.py` respects structural boundaries like `(a)`, `(b)` and targets 300–700 tokens per chunk with a hard cap of 1800.

**Indexing/Retrieval (`rag.py`):** `TaxCodeRAG` class wraps LlamaIndex + ChromaDB. Embeddings are truncated at 1800 tokens before sending to Ollama. Retrieves top-10 chunks by default. Indexes are persisted to `data/index_<name>/`.

**Query handling (`query.py`):** `handle_query()` runs LLM-based query expansion before retrieval (20s timeout), then formats retrieved chunks with section identifiers and hierarchy metadata.

**Web layer (`app.py`, `templates/index.html`):** Flask app with a single `POST /api/query` endpoint. The RAG instance is lazily initialized on first request via `get_rag()`.

### Key Defaults (in `rag.py`)
| Setting | Value |
|---|---|
| Embedding model | `nomic-custom` |
| LLM | `llama3.1:8b` |
| Chunks file | `data/rag_chunks2.json` |
| Index dir | `data/index_rag_chunks2/` |
| Retrieval top-k | 10 |
| Max embedding tokens | 1850 (truncated to 1800) |

### Chunk JSON Schema
```json
{
  "id": "section-162-a-1",
  "text": "...",
  "metadata": {
    "identifier": "162",
    "identifiers": ["162", "162(a)", "162(a)(1)"],
    "heading": "Trade or business expenses",
    "tag": "subsection",
    "token_count": 450
  },
  "parent_id": "section-162-a",
  "children_ids": [...]
}
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `RAG_INDEX_NAME` | Override index name |
| `RAG_CHUNKS_FILE` | Override chunks file path |
