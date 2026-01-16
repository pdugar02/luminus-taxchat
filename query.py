"""
RAG query logic for the Tax Code web app.
Provides get_rag and request handlers used by app.py.
"""

from index import TaxCodeRAG
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


def handle_query(data: dict) -> tuple[dict, int]:
    """Handle query requests and return (payload, status_code)."""
    try:
        question = data.get('question', '').strip()
        retrieve_only = data.get('retrieve_only', False)

        if not question:
            return {'error': 'Question is required'}, 400

        rag_system = get_rag()

        # Get optional parameters
        expand_query = data.get('expand_query', True)  # Enable query expansion by default
        top_k = data.get('top_k', 10)  # Number of chunks to retrieve

        # Query the RAG system with improved retrieval
        if retrieve_only:
            # Just retrieve chunks without LLM (uses improved retrieval with query expansion)
            result = rag_system.query(
                question,
                retrieve_only=True,
                expand_query=expand_query,
                top_k=top_k
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
                'retrieve_only': True
            }, 200
        else:
            # Full query with LLM (uses improved retrieval with query expansion)
            try:
                result = rag_system.query(
                    question,
                    include_sources=True,
                    expand_query=expand_query,
                    top_k=top_k
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
                    'chunks': chunks
                }, 200
            except Exception as e:
                # If LLM times out, still try to retrieve chunks
                try:
                    retrieve_result = rag_system.query(
                        question,
                        retrieve_only=True,
                        expand_query=expand_query,
                        top_k=top_k
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
                        'error': str(e)
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
