"""
RAG Index Builder and Query Interface for Tax Code
Uses LlamaIndex with local Llama 3.1 via Ollama
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tiktoken
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    Settings,
)
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

class TaxCodeRAG:
    """RAG system for querying the US Tax Code."""

    # Embedding model context limit (must match chunk cap so no chunk exceeds this).
    MAX_EMBEDDING_TOKENS = 1800
    # Safety margin: truncate to this many tokens so tokenizer differences don't exceed limit
    EMBEDDING_TRUNCATE_TOKENS = 1800
    
    def __init__(
        self,
        chunks_path: Optional[str] = None,
        index_dir: Optional[str] = None,
        embedding_model: str = "nomic-custom", # "nomic-embed-text", "embeddinggemma"
        ollama_model: str = "llama3.1:8b",  # Use the model tag you pulled (e.g., llama3.1:8b)
        ollama_base_url: str = "http://localhost:11434",
        auto_build: bool = True,
    ):
        """
        Initialize the RAG system.

        Args:
            chunks_path: Path to chunks json file
            index_dir: Directory to save/load index (default: ./data/index)
            embedding_model: Ollama model name for embeddings (default: nomic-custom)
            ollama_model: Ollama model name for LLM (default: llama3.1:8b)
                            Use the exact model name you pulled, including tags like :8b
            ollama_base_url: Ollama server URL
            auto_build: If True (default), auto-build or load the index on init.
                        Set False when the caller (e.g. build command) will trigger the build itself.
        """
        data_dir = Path(__file__).parent / "data"
        self.chunks_path = Path(chunks_path) if chunks_path else data_dir / "rag_chunks2.json"
        self.index_dir = Path(index_dir) if index_dir else data_dir / "index_chroma"
        # Resolve to absolute path so Chroma always uses the same DB regardless of process cwd
        if not self.index_dir.is_absolute():
            self.index_dir = (Path(__file__).parent / self.index_dir).resolve()
        else:
            self.index_dir = self.index_dir.resolve()
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        # Token counter for truncation (embedding/context limits). cl100k_base is fast and
        # sufficient for approximate limits; no need to match a specific LLM.
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
    
        # Initialize embedding model (using nomic-embed-text)
        print(f"Using Ollama for embeddings: {embedding_model}")
        Settings.embed_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url=ollama_base_url
        )

        # Initialize LLM (Ollama)
        print(f"Initializing Ollama LLM: {ollama_model}")
        Settings.llm = Ollama(model=ollama_model, base_url=ollama_base_url, request_timeout=120.0)
        

        # Chroma stores the index in index_dir. PersistentClient opens existing DB if present.
        # Load vs build is decided in _load_or_build_index (load if collection has vectors, else build).
        chroma_path = self.index_dir
        chroma_path.mkdir(parents=True, exist_ok=True)
        chroma_path_str = str(chroma_path)
        print(f"Chroma DB path (absolute): {chroma_path_str}")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path_str)

        self.index = None
        self.retriever = None
        self.query_engine = None

        if auto_build:
            self._init_index()

    def _init_index(self):
        """Load or build the index and wire up the retriever/query engine."""
        self.index = self._load_or_build_index()
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10,
        )
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever,
        )

    def _load_chunks(self) -> List[Dict]:
        """Load chunks from JSON file."""
        print(f"Loading chunks from {self.chunks_path}")
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks")
        return chunks
    
    def _get_node_text(self, node) -> str:
        """Helper to get full text from a node (original_text if available, else node.text)."""
        if hasattr(node, 'metadata') and node.metadata:
            return node.metadata.get('original_text') or node.text
        return node.text
    
    def format_source(self, node, preview_length: int = 300) -> Dict:
        """Format a node as a source dictionary."""
        metadata = node.metadata if hasattr(node, 'metadata') else {}
        ident = metadata.get('identifier')
        identifiers = metadata.get('identifiers')
        if identifiers is None and ident is not None:
            identifiers = [ident]
        return {
            'id': node.node_id,
            'text': self._get_node_text(node),
            'text_preview': node.text[:preview_length] + '...' if len(node.text) > preview_length else node.text,
            'score': getattr(node, 'score', None),
            'metadata': {
                'identifier': ident,
                'identifiers': identifiers or [],
                'heading': metadata.get('heading'),
                'tag': metadata.get('tag'),
            } if metadata else {}
        }
    
    def _truncate_text(self, text: str, max_tokens: int, chunk_id: str = None) -> Tuple[str, Dict]:
        """
        Truncate text to fit within token limit, preserving sentence boundaries.
        
        Returns:
            (truncated_text, metadata_dict)
        """
        tokens = self.token_encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text, {}
        
        # Truncate to max tokens
        truncated_tokens = tokens[:max_tokens]
        try:
            truncated_text = self.token_encoder.decode(truncated_tokens)
        except Exception:
            # Fallback to character-based truncation
            truncated_text = text[:max_tokens * 4]
        
        # Try to end at sentence boundary
        for delimiter in ['.', '\n']:
            pos = truncated_text.rfind(delimiter)
            if pos > len(truncated_text) * 0.9:
                truncated_text = truncated_text[:pos + 1]
                break
        
        # Validate UTF-8
        try:
            truncated_text.encode('utf-8')
        except UnicodeEncodeError:
            truncated_text = truncated_text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        
        metadata = {
            'truncated': True,
            'original_token_count': len(tokens),
            'truncated_token_count': len(self.token_encoder.encode(truncated_text))
        }
        
        return truncated_text, metadata
    
    def _chunk_to_node(self, chunk: Dict) -> TextNode:
        """Convert a chunk dictionary to a LlamaIndex TextNode."""
        original_text = chunk.get('text', '')
        metadata = chunk.get('metadata', {})
        chunk_id = chunk.get('id', 'unknown')
        
        # Keep metadata minimal: LlamaIndex embeds metadata_str + "\n\n" + text, so long
        # lists (identifiers, children_ids) blow the embedding context window (e.g. nomic-embed-text).
        # Store only short, scalar fields; format_source derives identifiers from identifier when needed.
        node_metadata = {
            'id': chunk_id,
            'tag': metadata.get('tag'),
            'identifier': metadata.get('identifier'),
            'heading': (metadata.get('heading') or '')[:100],  # cap heading length
        }
        
        # Build prefix from identifier(s) and heading
        prefix_parts = []
        ids = metadata.get('identifiers') or ([metadata.get('identifier')] if metadata.get('identifier') is not None else [])
        if ids:
            prefix_parts.append('§' + ', §'.join(str(i) for i in ids))
        if metadata.get('heading'):
            prefix_parts.append(metadata.get('heading'))
        prefix = f"{' '.join(prefix_parts)}: " if prefix_parts else ""
        
        # Calculate available tokens for text (reserve for prefix + marker)
        prefix_tokens = len(self.token_encoder.encode(prefix)) if prefix else 0
        reserved_tokens = prefix_tokens
        
        # Build full text with prefix first to check total size
        full_text_with_prefix = prefix + original_text
        full_tokens = self.token_encoder.encode(full_text_with_prefix)
        
        # Check if we need truncation
        limit = getattr(self, "_embedding_context_limit", self.MAX_EMBEDDING_TOKENS)
        if len(full_tokens) > limit:
            # Need to truncate - calculate how much of original_text we can keep
            # Account for prefix + marker
            max_text_tokens = limit - reserved_tokens
            
            # Truncate original text
            truncated_text, trunc_meta = self._truncate_text(original_text, max_text_tokens, chunk_id)
            
            # Build final text
            text = prefix + truncated_text
            
            # CRITICAL: Re-check final token count and truncate more if needed
            final_tokens = self.token_encoder.encode(text)
            if len(final_tokens) > limit:
                # Still too long - need more aggressive truncation
                # Remove excess tokens from the truncated_text
                excess = len(final_tokens) - limit
                truncated_text_tokens = self.token_encoder.encode(truncated_text)
                
                if len(truncated_text_tokens) > excess:
                    # Remove excess tokens
                    truncated_text_tokens = truncated_text_tokens[:len(truncated_text_tokens) - excess]
                    truncated_text = self.token_encoder.decode(truncated_text_tokens)
                    text = prefix + truncated_text
        else:
            # Text fits without truncation
            text = full_text_with_prefix

        # Exclude ALL metadata from embedding: LlamaIndex sends metadata_str + "\n\n" + text to
        # the embed model. Excluding metadata means only node.text (already token-capped) is sent,
        # giving exact control over the embedding input size.
        return TextNode(
            text=text,
            metadata=node_metadata,
            id_=chunk_id,
            excluded_embed_metadata_keys=list(node_metadata.keys()),
            excluded_llm_metadata_keys=["identifiers", "children_ids"],
        )
    
    def _build_index(self) -> VectorStoreIndex:
        """Build index from chunks."""
        print("Building index from chunks...")
        chunks = self._load_chunks()
        
        # Convert chunks to nodes with progress tracking (use model-specific limit for nomic)
        context_limit = getattr(self, "_embedding_context_limit", self.MAX_EMBEDDING_TOKENS)
        truncate_to = getattr(self, "_embedding_truncate_to", self.EMBEDDING_TRUNCATE_TOKENS)
        nodes = []
        for i, chunk in enumerate(chunks):
            try:
                node = self._chunk_to_node(chunk)
                # Enforce hard cap so embedding model never sees over-long input
                final_tokens = self.token_encoder.encode(node.text)
                input_tokens = len(final_tokens)
                original_tokens = input_tokens
                truncated = False
                if input_tokens > truncate_to:
                    truncated_text = self.token_encoder.decode(final_tokens[:truncate_to])
                    meta = dict(node.metadata) if node.metadata else {}
                    meta['hard_truncated'] = True
                    meta['original_token_count'] = input_tokens
                    node = TextNode(
                        text=truncated_text,
                        metadata=meta,
                        id_=node.node_id,
                        excluded_embed_metadata_keys=list(node.excluded_embed_metadata_keys or []),
                        excluded_llm_metadata_keys=list(node.excluded_llm_metadata_keys or []),
                    )
                    input_tokens = truncate_to
                    truncated = True
                # Debug: print first chunk and any truncated chunk (input_length vs context_limit)
                if i == 0 or truncated:
                    msg = f"  Chunk {i}: id={node.node_id} | input_tokens={input_tokens} | context_limit={context_limit}"
                    if truncated:
                        msg += f" (was {original_tokens} before truncation)"
                    print(msg)
                    print(node.metadata)
                nodes.append(node)
            except Exception as e:
                print(f"ERROR processing chunk {chunk.get('id', 'unknown')}: {e}")
                continue
        
        print(f"Converted {len(nodes)} chunks to nodes")
        
        # Pre-check: find nodes that exceed embed context limit (same string LlamaIndex sends to embed model)
        embed_limit = getattr(self, "_embedding_context_limit", self.MAX_EMBEDDING_TOKENS)
        over_limit = []
        for node in nodes:
            embed_content = node.get_content(metadata_mode=MetadataMode.EMBED)
            n_tokens = len(self.token_encoder.encode(embed_content))
            if n_tokens > embed_limit:
                over_limit.append((node.node_id, n_tokens, len(embed_content)))
        if over_limit:
            print(f"\n*** {len(over_limit)} node(s) exceed embed context limit ({embed_limit} tokens):")
            for nid, tokens, chars in sorted(over_limit, key=lambda x: -x[1])[:20]:
                print(f"    node_id={nid}  tokens={tokens}  chars={chars}")
            if len(over_limit) > 20:
                print(f"    ... and {len(over_limit) - 20} more")
            print("*** Fix truncation in _chunk_to_node or increase limit, then rebuild.\n")
        
        # Chroma: create collection and empty vector store, then index inserts nodes (with embeddings)
        chroma_collection = self.chroma_client.get_or_create_collection("tax_code_rag")
        vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection,
            client=self.chroma_client,
        )
        index = VectorStoreIndex(
            nodes=nodes,
            vector_store=vector_store,
            embed_model=Settings.embed_model,
            show_progress=True,
        )
        # Chroma already persists to disk via PersistentClient; no storage_context.persist needed
        print(f"Index saved to Chroma ({self.index_dir})")
        return index
    
    def _load_or_build_index(self) -> VectorStoreIndex:
        """Load existing Chroma index, or build from scratch if the collection is empty."""
        chroma_collection = self.chroma_client.get_or_create_collection("tax_code_rag")
        print(chroma_collection)
        print(self.chroma_client)
        if chroma_collection.count() == 0:
            print("No existing Chroma index. Building new index...")
            return self._build_index()
        print(f"Loading existing Chroma index ({chroma_collection.count()} vectors)")
        vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection,
            client=self.chroma_client,
        )
        return VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=Settings.embed_model,
        )

    @staticmethod
    def get_index_dir_from_chunks(chunks_file: str, index_name: str = None) -> Path:
        """
        Determine index directory path from chunks file or index name.
        
        Args:
            chunks_file: Path to chunks JSON file
            index_name: Optional index name (if None, derived from chunks filename)
            
        Returns:
            Path to index directory
        """
        data_dir = Path(__file__).parent / "data"
        
        if index_name:
            return data_dir / f"index_{index_name}"
        else:
            # Derive from chunks filename (e.g., rag_chunks2.json -> index_rag_chunks2)
            chunks_path = Path(chunks_file)
            chunks_stem = chunks_path.stem  # e.g., "rag_chunks2"
            return data_dir / f"index_{chunks_stem}"
    
    @staticmethod
    def check_ollama(ollama_base_url: str = "http://localhost:11434") -> bool:
        """
        Check if Ollama is running and accessible.
        """
        try:
            import requests
            response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


def list_indexes():
    """List all available indexes."""
    data_dir = Path(__file__).parent / "data"
    indexes = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("index_")]
    
    print("="*80)
    print("Available Indexes")
    print("="*80)
    
    if not indexes:
        print("No indexes found.")
        return
    
    for idx_dir in sorted(indexes):
        index_name = idx_dir.name.replace("index_", "")
        size = sum(f.stat().st_size for f in idx_dir.rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)
        print(f"  {index_name:30s}  {size_mb:6.1f} MB  ({idx_dir})")
    
    print("="*80)


def build_index_cmd(
    chunks_file: str,
    index_name: str = None,
    embedding_model: str = "nomic-custom", # "nomic-embed-text", "embeddinggemma"
    ollama_base_url: str = "http://localhost:11434",
    force_rebuild: bool = False
):
    """Build a vector index from a chunks file."""
    chunks_path = Path(chunks_file)
    if not chunks_path.exists():
        print(f"Error: Chunks file not found: {chunks_file}")
        return
    
    # Use TaxCodeRAG method to determine index directory (resolve to absolute for consistency)
    index_dir = TaxCodeRAG.get_index_dir_from_chunks(chunks_file, index_name)
    index_dir = index_dir.resolve()

    print("="*80)
    print("Building Index")
    print("="*80)
    print(f"Chunks file: {chunks_path}")
    print(f"Index directory: {index_dir}")
    print(f"Embedding model: {embedding_model}")
    print("="*80)
    
    # Check if Ollama is running
    if not TaxCodeRAG.check_ollama(ollama_base_url):
        print("Error: Could not connect to Ollama.")
        print("Make sure Ollama is installed and running.")
        print("Install: https://ollama.ai")
        print(f"Then run: ollama pull {embedding_model}")
        return
    
    # Initialize RAG (auto_build=False so we control when the build happens below)
    print("\nInitializing RAG system...")
    rag = TaxCodeRAG(
        chunks_path=str(chunks_path),
        index_dir=str(index_dir),
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        auto_build=False,
    )

    # Decide whether to build
    existing_count = rag.chroma_client.get_or_create_collection("tax_code_rag").count()
    if force_rebuild or existing_count == 0:
        if force_rebuild and existing_count > 0:
            print(f"\nForce-rebuilding (existing index has {existing_count} vectors)...")
        else:
            print("\nNo existing index found. Building...")
        try:
            rag.chroma_client.delete_collection("tax_code_rag")
        except Exception:
            pass
        rag._init_index()
    else:
        print(f"\n✓ Index already exists ({existing_count} vectors). Use --force to rebuild.")
        rag._init_index()
    
    print("\n" + "="*80)
    print("Index Build Complete!")
    print("="*80)
    print(f"Index location: {index_dir}")
    index_name_for_app = index_dir.name.replace("index_", "")
    print("\nTo use this index in the app, specify:")
    print(f"  --index-name {index_name_for_app}")
    print(f"  or RAG_INDEX_NAME={index_name_for_app}")
    print("="*80)


def main():
    """Main function with subcommands for building and managing indexes."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Tax Code RAG System - Build and manage vector indexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python rag.py build
        python rag.py build data/rag_chunks2.json
        python rag.py build data/rag_chunks2.json --index-name my_index --force
        python rag.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run', required=True)
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build a vector index from chunks')
    build_parser.add_argument(
        'chunks_file',
        nargs='?',
        default='data/rag_chunks2.json',
        help='Path to chunks JSON file (default: data/rag_chunks2.json)',
    )
    build_parser.add_argument('--index-name', type=str, default='rag_chunks2', help='Name for the index directory')
    build_parser.add_argument('--embedding-model', type=str, default='nomic-custom', help='Ollama embedding model')
    build_parser.add_argument('--ollama-base-url', type=str, default='http://localhost:11434', help='Ollama server URL')
    build_parser.add_argument('--force', action='store_true', help='Force rebuild even if index exists')
    
    # List command
    subparsers.add_parser('list', help='List all available indexes')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        build_index_cmd(
            chunks_file=args.chunks_file,
            index_name=args.index_name,
            embedding_model=args.embedding_model,
            ollama_base_url=args.ollama_base_url,
            force_rebuild=args.force
        )
    elif args.command == 'list':
        list_indexes()


if __name__ == "__main__":
    main()

