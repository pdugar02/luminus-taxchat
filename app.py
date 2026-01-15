"""
Flask web application for Tax Code RAG System
"""

from flask import Flask, render_template, request, jsonify
from index import TaxCodeRAG
import traceback

app = Flask(__name__)

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
            # Default: use rag_chunks2
            index_dir = Path(__file__).parent / "data" / "index_rag_chunks2"
            chunks_file = str(Path(__file__).parent / "data" / "rag_chunks2.json")
        
        rag = TaxCodeRAG(
            chunks_path=chunks_file,
            index_dir=str(index_dir)
        )
    return rag

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    """Handle query requests."""
    try:
        data = request.json
        question = data.get('question', '').strip()
        retrieve_only = data.get('retrieve_only', False)
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
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
            
            return jsonify({
                'answer': result.get('answer', ''),
                'chunks': chunks,
                'retrieve_only': True
            })
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
                
                return jsonify({
                    'answer': result.get('answer', ''),
                    'sources': result.get('sources', []),
                    'chunks': chunks
                })
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
                    return jsonify({
                        'answer': f'Retrieved relevant chunks, but LLM generation failed: {str(e)}',
                        'sources': [],
                        'chunks': chunks,
                        'error': str(e)
                    })
                except Exception as e2:
                    return jsonify({
                        'error': f'Both retrieval and LLM failed: {str(e2)}'
                    }), 500
            
    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        rag_system = get_rag()
        return jsonify({
            'status': 'healthy',
            'index_loaded': rag_system.index is not None
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    import sys
    import os
    
    # Check for index name from environment variable or command line
    index_name = os.getenv('RAG_INDEX_NAME', None)
    chunks_file = os.getenv('RAG_CHUNKS_FILE', None)
    
    # Parse command line arguments
    port = 5001
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--index-name' and i + 1 < len(sys.argv):
            index_name = sys.argv[i + 1]
        elif arg == '--chunks-file' and i + 1 < len(sys.argv):
            chunks_file = sys.argv[i + 1]
        elif arg == '--port' and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
        elif arg.isdigit():
            # Legacy: port as first positional argument
            port = int(arg)
    
    # Initialize RAG with specified index
    if index_name or chunks_file:
        print(f"\nUsing index: {index_name or 'from chunks file'}")
        get_rag(index_name=index_name, chunks_file=chunks_file)
    
    print("\n" + "="*80)
    print("Starting Flask server...")
    if index_name:
        print(f"Using index: {index_name}")
    print(f"Open http://localhost:{port} in your browser")
    print("="*80 + "\n")
    print("Usage:")
    print("  python app.py --index-name <name> --port <port>")
    print("  RAG_INDEX_NAME=<name> python app.py")
    print("="*80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=port)

