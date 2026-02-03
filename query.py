"""
RAG query logic for the Tax Code web app.
Provides get_rag and request handlers used by app.py.
"""

import json
from typing import List

from rag import TaxCodeRAG
from llama_index.llms.ollama import Ollama
import traceback

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


def query(rag: TaxCodeRAG,
        question: str,
        include_sources: bool = True,
        retrieve_only: bool = False,
        expand: bool = True,
        top_k: int = 10,
    ) -> dict:
        """
        Query the tax code.

        Args:
            question: User's question about the tax code
            include_sources: Whether to include source citations
            retrieve_only: If True, only retrieve relevant chunks without LLM generation (faster for testing)
            expand: Whether to expand/rewrite the query using LLM (default: True)
            top_k: Number of chunks to retrieve (default: 10)

        Returns:
            Dictionary with 'answer' and optionally 'sources'
        """
        print(f"\nOriginal Query: {question}")

        
        # Queries to use for retrieval
        queries = reframe_query(question)


        # Query the index (includes LLM generation - this is where timeouts happen)
        import time
        start = time.time()

        # Embedding search: one retrieval per query, then merge by node_id (keep higher score)
        node_by_id = {}
        for q in queries:
            if not q:
                continue
            expanded_q = expand_query(q) if expand else q
            nodes = rag.retriever.retrieve(expanded_q)[:top_k]
            for node in nodes:
                nid = getattr(node, "node_id", None)
                if not nid:
                    continue
                existing = node_by_id.get(nid)
                if existing is None:
                    node_by_id[nid] = node
                else:
                    existing_score = getattr(existing, "score", None)
                    node_score = getattr(node, "score", None)
                    if existing_score is None and node_score is not None:
                        node_by_id[nid] = node
                    elif (
                        existing_score is not None
                        and node_score is not None
                        and node_score > existing_score
                    ):
                        node_by_id[nid] = node
        merged = list(node_by_id.values())
        merged.sort(key=lambda n: getattr(n, "score", None) or float("-inf"), reverse=True)
        retrieved_nodes = merged[:top_k]
        retrieval_time = time.time() - start
        print(f"\n[Retrieval] {retrieval_time:.3f}s — {len(retrieved_nodes)} chunks")
        print("="*80)

        if retrieve_only: # return retrieved chunks without trying to formulate an answer
            return {
                'answer': f"Retrieved {len(retrieved_nodes)} relevant chunks (retrieve_only mode)",
                'sources': [rag._format_source(node) for node in retrieved_nodes]
            }
        
        # Then generate answer with LLM (this can be slow and cause timeouts)
        print("\nGenerating answer with LLM...")
        try:
            gen_start = time.time()
            response = rag.query_engine.query(question)
            generation_time = time.time() - gen_start
            print(f"[Generation] {generation_time:.3f}s")
            total_time = time.time() - start
            print(f"[Total] {total_time:.3f}s (retrieval: {retrieval_time:.3f}s, generation: {generation_time:.3f}s)")
        except Exception as e:
            print(f"\n⚠️  LLM generation failed: {e}")
            print("But we successfully retrieved the chunks above - the embedding search is working!")
            # Return the retrieved chunks even if LLM fails
            return {
                'answer': f"LLM generation timed out, but retrieved {len(retrieved_nodes)} relevant chunks. See retrieved chunks above.",
                'sources': [rag._format_source(node) for node in retrieved_nodes]
            }
        
        # Extract answer
        result = {'answer': str(response), 'sources': [rag._format_source(node, preview_length=200) for node in response.source_nodes[:5]]}
        return result
    

def handle_query(data: dict) -> tuple[dict, int]:
    """Handle query requests and return (payload, status_code)."""
    question = data.get('question', '').strip()
    if not question:
        return {'error': 'Question is required', 'sources': []}, 400

    rag_system = get_rag()
    query_engine = rag_system.index.as_query_engine(
        include_text=True, response_mode="tree_summarize", retriever_mode='hybrid'
    )
    result = query_engine.query(question)

    return {'answer': result.response, 'sources': [rag._format_source(node, preview_length=200) for node in result.source_nodes[:5]]}, 200