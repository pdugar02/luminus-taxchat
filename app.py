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

def get_rag():
    """Lazy initialization of RAG system."""
    global rag
    if rag is None:
        rag = TaxCodeRAG()
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
    # Use port 5001 by default (5000 is often used by AirPlay on macOS)
    port = 5001
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default port 5001.")
    
    print("\n" + "="*80)
    print("Starting Flask server...")
    print(f"Open http://localhost:{port} in your browser")
    print("="*80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=port)

