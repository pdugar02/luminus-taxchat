"""
RAG query logic for the Tax Code web app.
Provides get_rag and request handlers used by app.py.
"""

import json
from typing import List

from rag import TaxCodeRAG
from llama_index.llms.ollama import Ollama
import time

rag = None
REFRAME_PROMPT = """You are a precise legal research assistant.
Rephrase the user query into 3 alternative search queries that preserve the original meaning,
use legal/tax terminology where helpful, and avoid adding new facts.

Return ONLY a JSON array of exactly 3 strings. No extra text.

User query:
{question}
"""

EXPANSION_PROMPT = """
You are helping to improve a search query for finding information in the US Tax Code (Title 26).

Original question: {question}

Generate an improved search query that:
1. Includes key tax code terminology (e.g., "taxable income", "filing status", "tax brackets", "standard deduction")
2. Expands abbreviations (e.g., "MFS" -> "married filing separately")
3. Adds related terms (e.g., "tax rate" for "how much am I taxed")
4. Keeps the core intent of the question
5. If a specific section is asked for in the original query, make sure to include that section number
6. Evaluate key phrases and terms in the search query. If there is something ambiguous, rephrase it with your best interpretation of what the user is trying to ask given the context.
7. Focus on the terms that actually matter to the user's question. If the user is asking about a specific section, focus on that section. If the user is asking about a specific term, focus on that term. Do not focus on arbitrary terms or numbers that are not going to be present in the most relevant chunks to the query.

Return ONLY the improved search query, nothing else:
"""


def get_rag(index_name: str = None, chunks_file: str = None):
    """
    Lazy initialization of RAG system.

    Args:
        index_name: Name of the index to use (e.g., 'rag_chunks2' for index_rag_chunks2)
                    If None, uses default from chunks_file or rag_chunks2.json
        chunks_file: Path to chunks file (only used if index doesn't exist and needs building)
    """
    global rag
    if rag is None:
        from pathlib import Path

        # Determine index directory
        if index_name:
            index_dir = Path(__file__).parent / "data" / f"index_{index_name}"
        elif chunks_file:
            # Derive from chunks filename
            chunks_path = Path(chunks_file)
            index_dir = Path(__file__).parent / "data" / f"index_{chunks_path.stem}"
        else:
            # Default: use rag_chunks2, with legacy fallback to data/index
            data_dir = Path(__file__).parent / "data"
            default_index_dir = data_dir / "index_rag_chunks2"
            legacy_index_dir = data_dir / "index"
            if default_index_dir.exists() or not legacy_index_dir.exists():
                index_dir = default_index_dir
            else:
                index_dir = legacy_index_dir
            chunks_file = str(data_dir / "rag_chunks2.json")

        rag = TaxCodeRAG(
            chunks_path=chunks_file,
            index_dir=str(index_dir)
        )
    return rag

def reframe_query(question: str, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434") -> List[str]:
    """Generate three re-phrased queries using Llama 3.1."""
    llm = Ollama(model=model, base_url=base_url, request_timeout=20.0)
    prompt = REFRAME_PROMPT.format(question=question)
    response = llm.complete(prompt)
    raw = response.text.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed][:3]
    except json.JSONDecodeError:
        pass
    # Fallback: split lines and take first three non-empty lines
    lines = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
    return lines[:3]

def expand_query(question: str) -> str:
        """
        Expand/rewrite the query to improve retrieval.
        Uses LLM to generate a better search query with related terms.
        """
        llm = Ollama(model="llama3.1:8b", base_url="http://localhost:11434", request_timeout=20.0)
        expanded = llm.complete(EXPANSION_PROMPT.format(question=question)).text.strip()
        return expanded
    

def handle_query(data: dict) -> tuple[dict, int]:
    """Handle query requests and return (payload, status_code)."""
    question = data.get('question', '').strip()
    if not question:
        return {'error': 'Question is required', 'sources': []}, 400

    rag_system = get_rag()
    query_engine = rag_system.index.as_query_engine(
        include_text=True, response_mode="tree_summarize", retriever_mode='hybrid'
    )
    start = time.time()
    question = expand_query(question)
    result = query_engine.query(question)
    end = time.time()
    print(f"Time taken to query: {end - start} seconds")
    return {'answer': result.response, 'sources': [rag_system.format_source(node, preview_length=200) for node in result.source_nodes[:5]]}, 200