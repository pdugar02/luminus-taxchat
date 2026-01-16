"""
Flask web application for Tax Code RAG System.
Routes and server start live here; query logic lives in query.py.
"""

from flask import Flask, render_template, request, jsonify
from query import get_rag, handle_query, health_status

app = Flask(__name__)


@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    """Handle query requests."""
    payload, status = handle_query(request.json or {})
    return jsonify(payload), status


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    payload, status = health_status()
    return jsonify(payload), status


def run_app():
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


if __name__ == '__main__':
    run_app()

