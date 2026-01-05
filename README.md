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

### Requirements

```bash
pip install lxml
```
