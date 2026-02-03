"""
RAG query logic for the Tax Code web app.
Provides get_rag and request handlers used by app.py.
"""

import json
from typing import List

from rag import TaxCodeRAG
from llama_index.llms.ollama import Ollama
import traceback

# Initialize RAG system
print("Initializing RAG system...")
rag = None


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


REFRAME_PROMPT = """You are a precise legal research assistant.
Rephrase the user query into 3 alternative search queries that preserve the original meaning,
use legal/tax terminology where helpful, and avoid adding new facts.

Return ONLY a JSON array of exactly 3 strings. No extra text.

User query:
{question}
"""


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


def handle_query(data: dict) -> tuple[dict, int]:
    """Handle query requests and return (payload, status_code)."""
    try:
        question = data.get('question', '').strip()
        retrieve_only = data.get('retrieve_only', False)

        if not question:
            return {'error': 'Question is required'}, 400

        # Generate re-phrased queries for retrieval
        try:
            rephrased_queries = reframe_query(question)
        except Exception:
            rephrased_queries = []

        # Use original question + rephrases for retrieval
        retrieval_queries = []
        for q in [question] + rephrased_queries:
            if q and q not in retrieval_queries:
                retrieval_queries.append(q)

        rag_system = get_rag()

        # Get optional parameters
        expand_query = data.get('expand_query', True)  # Enable query expansion by default
        top_k = data.get('top_k', 5)  # Number of chunks to retrieve

        # Query the RAG system with improved retrieval
        if retrieve_only:
            # Just retrieve chunks without LLM (uses improved retrieval with query expansion)
            result = rag_system.query(
                question,
                retrieve_only=True,
                expand_query=expand_query,
                top_k=top_k,
                search_queries=retrieval_queries
            )

            chunks = []
            for source in result.get('sources', []):
                chunks.append({
                    'id': source.get('id'),
                    'text': source.get('text', source.get('text_preview', '')),  # Use full text, fallback to preview
                    'score': source.get('score'),
                    'metadata': source.get('metadata', {})
                })

            return {
                'answer': result.get('answer', ''),
                'chunks': chunks,
                'retrieve_only': True,
                'rephrased_queries': rephrased_queries
            }, 200
        else:
            # Full query with LLM (uses improved retrieval with query expansion)
            try:
                result = rag_system.query(
                    question,
                    include_sources=True,
                    expand_query=expand_query,
                    top_k=top_k,
                    search_queries=retrieval_queries
                )

                # Extract chunks from sources
                chunks = []
                for source in result.get('sources', []):
                    chunks.append({
                        'id': source.get('id'),
                        'text': source.get('text', source.get('text_preview', '')),  # Use full text, fallback to preview
                        'score': source.get('score'),
                        'metadata': source.get('metadata', {})
                    })

                return {
                    'answer': result.get('answer', ''),
                    'sources': result.get('sources', []),
                    'chunks': chunks,
                    'rephrased_queries': rephrased_queries
                }, 200
            except Exception as e:
                # If LLM times out, still try to retrieve chunks
                try:
                    retrieve_result = rag_system.query(
                        question,
                        retrieve_only=True,
                        expand_query=expand_query,
                        top_k=top_k,
                        search_queries=retrieval_queries
                    )
                    chunks = []
                    for source in retrieve_result.get('sources', []):
                        chunks.append({
                            'id': source.get('id'),
                            'text': source.get('text', source.get('text_preview', '')),  # Use full text, fallback to preview
                            'score': source.get('score'),
                            'metadata': source.get('metadata', {})
                        })
                    return {
                        'answer': f'Retrieved relevant chunks, but LLM generation failed: {str(e)}',
                        'sources': [],
                        'chunks': chunks,
                        'error': str(e),
                        'rephrased_queries': rephrased_queries
                    }, 200
                except Exception as e2:
                    return {
                        'error': f'Both retrieval and LLM failed: {str(e2)}'
                    }, 500

    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        return {'error': error_msg}, 500


def health_status() -> tuple[dict, int]:
    """Health check payload and status."""
    try:
        rag_system = get_rag()
        return {
            'status': 'healthy',
            'index_loaded': rag_system.index is not None
        }, 200
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }, 500
