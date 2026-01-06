# Tax Chat

Tax code chat assistant for Luminus Analytics. The goal is to give IRS employees a quick way to look up complex tax questions and receive answers based on the actual US Tax Code.

## XML Ingestion

This project includes tools for parsing and chunking the US Code Title 26 (Internal Revenue Code) XML file.

### Files

- `ingest.py`: Main XML parser that extracts hierarchical chunks from the XML file
- `chunk.py`: Chunking utilities for splitting large text chunks while preserving relationships
- `data/usc26.xml`: The source XML file (55MB)
- `data/chunks.json`: Raw parsed chunks with full hierarchy
- `data/rag_chunks.json`: Chunks optimized for RAG ingestion

### Usage

1. **Parse the XML file:**

   ```bash
   python ingest.py
   ```

   This will:
   - Parse the XML file and extract hierarchical chunks
   - Preserve parent/child relationships
   - Split large chunks into smaller ones suitable for RAG
   - Generate two output files in the `data/` folder:
     - `data/chunks.json`: Raw parsed chunks with full hierarchy
     - `data/rag_chunks.json`: Chunks optimized for RAG ingestion

### How It Works

1. **XML Parsing**: The parser uses `lxml` to parse the USLM (US Legislative Markup) XML format, extracting:
   - Sections, subsections, paragraphs, and other structural elements
   - Text content with proper hierarchy
   - Identifiers (section numbers, etc.)
   - Parent/child relationships

2. **Chunking**: Large chunks are split into smaller pieces (default 1000 characters) while:
   - Preserving parent/child relationships
   - Maintaining context through overlap
   - Keeping metadata for each chunk

3. **Output Format**: Each chunk includes:
   - `id`: Unique identifier
   - `text`: The text content
   - `metadata`: Additional information including:
     - `parent_id`: ID of parent chunk
     - `children_ids`: List of child chunk IDs
     - `element_type`: Type of XML element (section, subsection, etc.)
     - `identifier`: Section number or other identifier

## RAG Query System

The project includes a RAG (Retrieval-Augmented Generation) system for querying the tax code using LlamaIndex and local Llama 3.1 via Ollama.

### Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Install and start Ollama:**

   - Download from https://ollama.ai
   - Pull the Llama 3.1 model:
     ```bash
     ollama pull llama3.1
     ```
   - Make sure Ollama is running (it should start automatically)

3. **Build the index:**

   The first time you run the query system, it will automatically build an index from `data/rag_chunks.json`. This may take several minutes as it creates embeddings for all chunks.

### Usage

**Interactive mode:**

```bash
python index.py
```

**Single query:**

```bash
python index.py "What is the tax rate for married couples filing jointly?"
```

### How It Works

1. **Indexing**: The system loads chunks from `data/rag_chunks.json` and creates embeddings using a HuggingFace model (BAAI/bge-small-en-v1.5 by default).

2. **Querying**: When you ask a question:
   - The system finds the most relevant chunks using semantic search
   - Retrieves top 5 relevant chunks with their metadata (section numbers, headings)
   - Formats context with section identifiers and parent/child relationships
   - Sends to Llama 3.1 via Ollama for answer generation
   - Returns answer with source citations

3. **Metadata Context**: The system includes section numbers, headings, and hierarchical relationships in the context to provide better answers.

### Configuration

You can customize the system by modifying parameters in `index.py`:

- `embedding_model`: Change the embedding model (default: "BAAI/bge-small-en-v1.5")
- `ollama_model`: Change the Ollama model (default: "llama3.1")
- `ollama_base_url`: Change Ollama server URL (default: "http://localhost:11434")
- `similarity_top_k`: Number of chunks to retrieve (default: 5)

### Rebuilding the Index

If you update the chunks, rebuild the index:

```python
from index import TaxCodeRAG
rag = TaxCodeRAG()
rag.rebuild_index()
```

### Future Migration to Vector Database

The system is designed for easy migration to a vector database (Chroma, Qdrant, etc.):

1. Install the vector database client (e.g., `pip install chromadb`)
2. Replace `SimpleVectorStore` with `ChromaVectorStore` or `QdrantVectorStore` in `index.py`
3. No changes needed to query logic

### Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:
- `lxml`: XML parsing
- `llama-index`: RAG framework
- `llama-index-embeddings-huggingface`: Embeddings
- `ollama`: Local LLM inference
