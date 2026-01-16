"""
RAG Index Builder and Query Interface for Tax Code
Uses LlamaIndex with local Llama 3.1 via Ollama
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tiktoken

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


class TaxCodeRAG:
    """RAG system for querying the US Tax Code."""
    
    # nomic-embed-text has a context window of 8192 tokens
    # Using conservative limit to avoid context length errors
    MAX_EMBEDDING_TOKENS = 7500  # Conservative limit (7500 out of 8192) to account for model overhead
    
    def __init__(
        self,
        chunks_path: Optional[str] = None,
        index_dir: Optional[str] = None,
        embedding_model: str = "nomic-embed-text",
        ollama_model: str = "llama3.1:8b",  # Use the model tag you pulled (e.g., llama3.1:8b)
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the RAG system.
        
        Args:
            chunks_path: Path to rag_chunks.json file
            index_dir: Directory to save/load index (default: ./data/index)
            embedding_model: Ollama model name for embeddings (default: nomic-embed-text)
            ollama_model: Ollama model name for LLM (default: llama3.1:8b)
                            Use the exact model name you pulled, including tags like :8b
            ollama_base_url: Ollama server URL
        """
        data_dir = Path(__file__).parent / "data"
        self.chunks_path = Path(chunks_path) if chunks_path else data_dir / "rag_chunks2.json"
        if index_dir:
            self.index_dir = Path(index_dir)
        else:
            # Prefer the newer named index directory, but fall back to legacy `data/index`
            default_index_dir = data_dir / "index_rag_chunks2"
            legacy_index_dir = data_dir / "index"
            if default_index_dir.exists() or not legacy_index_dir.exists():
                self.index_dir = default_index_dir
            else:
                self.index_dir = legacy_index_dir
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        
        # Initialize tiktoken encoder for token counting
        try:
            self.token_encoder = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            # Fallback to cl100k_base (used by GPT-4 and similar models)
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
        
        # Load or build index (auto_build=True for backward compatibility)
        # Set auto_build=False if you want to require pre-built indexes
        self.index = self._load_or_build_index(auto_build=True)
        
        # Create custom retriever with better configuration
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=20,  # Retrieve more candidates initially
        )
        
        # Create query engine directly from retriever (avoids conflict with as_query_engine)
        # Use "default" response mode (faster, single LLM call) instead of "compact" (multiple calls)
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
    
    def _format_source(self, node, preview_length: int = 300) -> Dict:
        """Format a node as a source dictionary."""
        metadata = node.metadata if hasattr(node, 'metadata') else {}
        return {
            'id': node.node_id,
            'text': self._get_node_text(node),
            'text_preview': node.text[:preview_length] + '...' if len(node.text) > preview_length else node.text,
            'score': getattr(node, 'score', None),
            'metadata': {
                'identifier': metadata.get('identifier'),
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
        
        # Build metadata
        node_metadata = {
            'id': chunk_id,
            'tag': metadata.get('tag'),
            'identifier': metadata.get('identifier'),
            'heading': metadata.get('heading'),
            'parent_id': metadata.get('parent_id'),
            'children_ids': metadata.get('children_ids', []),
            'element_id': metadata.get('element_id'),
            'chunk_index': metadata.get('chunk_index'),
            'original_text': original_text,
        }
        
        # Build prefix from identifier and heading
        prefix_parts = []
        if metadata.get('identifier'):
            prefix_parts.append(f"§{metadata.get('identifier')}")
        if metadata.get('heading'):
            prefix_parts.append(metadata.get('heading'))
        prefix = f"{' '.join(prefix_parts)}: " if prefix_parts else ""
        
        # Calculate available tokens for text (reserve for prefix + marker)
        truncation_marker = "\n\n[Text truncated for embedding]"
        prefix_tokens = len(self.token_encoder.encode(prefix)) if prefix else 0
        marker_tokens = len(self.token_encoder.encode(truncation_marker))
        reserved_tokens = prefix_tokens + marker_tokens
        
        # Build full text with prefix first to check total size
        full_text_with_prefix = prefix + original_text
        full_tokens = self.token_encoder.encode(full_text_with_prefix)
        
        # Check if we need truncation
        if len(full_tokens) > self.MAX_EMBEDDING_TOKENS:
            # Need to truncate - calculate how much of original_text we can keep
            # Account for prefix + marker
            max_text_tokens = self.MAX_EMBEDDING_TOKENS - reserved_tokens
            
            # Truncate original text
            truncated_text, trunc_meta = self._truncate_text(original_text, max_text_tokens, chunk_id)
            node_metadata.update(trunc_meta)
            
            # Build final text
            text = prefix + truncated_text + truncation_marker
            
            # CRITICAL: Re-check final token count and truncate more if needed
            final_tokens = self.token_encoder.encode(text)
            if len(final_tokens) > self.MAX_EMBEDDING_TOKENS:
                # Still too long - need more aggressive truncation
                # Remove excess tokens from the truncated_text
                excess = len(final_tokens) - self.MAX_EMBEDDING_TOKENS
                truncated_text_tokens = self.token_encoder.encode(truncated_text)
                
                if len(truncated_text_tokens) > excess:
                    # Remove excess tokens
                    truncated_text_tokens = truncated_text_tokens[:len(truncated_text_tokens) - excess]
                    truncated_text = self.token_encoder.decode(truncated_text_tokens)
                    text = prefix + truncated_text + truncation_marker
                    
                    # Verify again
                    final_tokens = self.token_encoder.encode(text)
                    if len(final_tokens) > self.MAX_EMBEDDING_TOKENS:
                        # Last resort: hard truncate the entire text (prefix + truncated + marker)
                        emergency_tokens = self.token_encoder.encode(text)[:self.MAX_EMBEDDING_TOKENS]
                        text = self.token_encoder.decode(emergency_tokens)
                        node_metadata['emergency_truncated'] = True
                else:
                    # Can't fit even without truncated_text - just use prefix (shouldn't happen)
                    text = prefix
                    node_metadata['emergency_truncated'] = True
        else:
            # Text fits without truncation
            text = full_text_with_prefix
        
        # Final validation: ensure UTF-8 and under limit
        try:
            text.encode('utf-8')
            final_count = len(self.token_encoder.encode(text))
            if final_count > self.MAX_EMBEDDING_TOKENS:
                # Absolute last resort
                print(f"WARNING: Chunk {chunk_id} still exceeds limit ({final_count} tokens). Hard truncating.")
                final_tokens = self.token_encoder.encode(text)[:self.MAX_EMBEDDING_TOKENS]
                text = self.token_encoder.decode(final_tokens)
                node_metadata['hard_truncated'] = True
        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            print(f"WARNING: Encoding error in chunk {chunk_id}: {e}. Fixing.")
            text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            node_metadata['encoding_fixed'] = True
        
        return TextNode(text=text, metadata=node_metadata, id_=chunk_id)
    
    def _build_index(self) -> VectorStoreIndex:
        """Build index from chunks."""
        print("Building index from chunks...")
        chunks = self._load_chunks()
        
        # Convert chunks to nodes with progress tracking
        nodes = []
        problematic_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                node = self._chunk_to_node(chunk)
                # Validate node text is within limits
                final_tokens = self.token_encoder.encode(node.text)
                if len(final_tokens) > self.MAX_EMBEDDING_TOKENS:
                    problematic_chunks.append((chunk.get('id', 'unknown'), len(final_tokens)))
                    print(f"WARNING: Chunk {chunk.get('id')} has {len(final_tokens)} tokens after processing")
                nodes.append(node)
            except Exception as e:
                print(f"ERROR processing chunk {chunk.get('id', 'unknown')}: {e}")
                # Skip problematic chunks to avoid crashing
                continue
        
        if problematic_chunks:
            print(f"\n⚠️  Found {len(problematic_chunks)} chunks that still exceed token limit")
            print("These will be hard-truncated during embedding generation")
        
        print(f"Converted {len(nodes)} chunks to nodes")
        
        # Create vector store (in-memory, but using LlamaIndex abstraction)
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Build index from nodes
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        
        # Save index
        print(f"Saving index to {self.index_dir}")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(self.index_dir))
        
        return index
    
    def _load_or_build_index(self, auto_build: bool = True) -> VectorStoreIndex:
        """
        Load existing index or build new one.
        
        Args:
            auto_build: If True, automatically build if index doesn't exist.
                       If False, raise error if index doesn't exist.
        """
        if self.index_dir.exists() and any(self.index_dir.iterdir()):
            try:
                print(f"Loading existing index from {self.index_dir}")
                storage_context = StorageContext.from_defaults(persist_dir=str(self.index_dir))
                index = load_index_from_storage(storage_context)
                print("Index loaded successfully")
                return index
            except Exception as e:
                print(f"Failed to load index: {e}")
                if auto_build:
                    print("Building new index...")
                    return self._build_index()
                else:
                    raise FileNotFoundError(
                        f"Index not found at {self.index_dir}. "
                        f"Please build it first using: python index.py build <chunks_file>"
                    ) from e
        else:
            if auto_build:
                print("No existing index found. Building new index...")
                return self._build_index()
            else:
                raise FileNotFoundError(
                    f"Index not found at {self.index_dir}. "
                    f"Please build it first using: python index.py build <chunks_file>"
                )
    
    def _expand_query(self, question: str) -> str:
        """
        Expand/rewrite the query to improve retrieval.
        Uses LLM to generate a better search query with related terms.
        """
        expansion_prompt = f"""You are helping to improve a search query for finding information in the US Tax Code (Title 26).

        Original question: {question}

        Generate an improved search query that:
        1. Includes key tax code terminology (e.g., "taxable income", "filing status", "tax brackets", "standard deduction")
        2. Expands abbreviations (e.g., "MFS" -> "married filing separately")
        3. Adds related terms (e.g., "tax rate" for "how much am I taxed")
        4. Keeps the core intent of the question

        Return ONLY the improved search query, nothing else:"""

        try:
            expanded = Settings.llm.complete(expansion_prompt).text.strip()
            print(f"Original query: {question}")
            print(f"Expanded query: {expanded}")
            return expanded
        except Exception as e:
            print(f"Query expansion failed: {e}. Using original query.")
            return question
    
    def _retrieve_with_metadata_filtering(
        self, 
        query: str, 
        top_k: int = 20,
        filter_sections: Optional[List[str]] = None
    ) -> List:
        """
        Retrieve chunks with optional metadata filtering.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            filter_sections: Optional list of section numbers to prioritize (e.g., ["1", "2"])
        """
        # Retrieve more candidates
        nodes = self.retriever.retrieve(query)
        
        # If we have section filters, boost those sections
        if filter_sections:
            filtered_nodes = []
            other_nodes = []
            
            for node in nodes:
                metadata = node.metadata if hasattr(node, 'metadata') else {}
                identifier = metadata.get('identifier', '')
                
                # Check if this node's section matches any filter
                matches = any(
                    identifier.startswith(sec) or sec in identifier 
                    for sec in filter_sections
                )
                
                if matches:
                    filtered_nodes.append(node)
                else:
                    other_nodes.append(node)
            
            # Prioritize filtered nodes, then add others
            nodes = filtered_nodes[:top_k] + other_nodes[:top_k - len(filtered_nodes)]
        else:
            nodes = nodes[:top_k]
        
        return nodes

    def _retrieve_multi_query(
        self,
        queries: List[str],
        top_k: int = 10,
        filter_sections: Optional[List[str]] = None
    ) -> List:
        """Retrieve and merge results across multiple queries."""
        node_by_id = {}

        for query in queries:
            if not query:
                continue
            nodes = self._retrieve_with_metadata_filtering(
                query,
                top_k=top_k,
                filter_sections=filter_sections
            )
            for node in nodes:
                node_id = getattr(node, "node_id", None)
                if not node_id:
                    continue
                existing = node_by_id.get(node_id)
                if existing is None:
                    node_by_id[node_id] = node
                else:
                    # Keep the node with the higher score if available
                    existing_score = getattr(existing, "score", None)
                    node_score = getattr(node, "score", None)
                    if existing_score is None and node_score is not None:
                        node_by_id[node_id] = node
                    elif (
                        existing_score is not None
                        and node_score is not None
                        and node_score > existing_score
                    ):
                        node_by_id[node_id] = node

        merged_nodes = list(node_by_id.values())

        def sort_key(n):
            score = getattr(n, "score", None)
            return score if score is not None else float("-inf")

        merged_nodes.sort(key=sort_key, reverse=True)
        return merged_nodes[:top_k]
    
    def query(
        self, 
        question: str, 
        include_sources: bool = True, 
        retrieve_only: bool = False,
        expand_query: bool = True,
        top_k: int = 10,
        search_queries: Optional[List[str]] = None
    ) -> Dict:
        """
        Query the tax code.
        
        Args:
            question: User's question about the tax code
            include_sources: Whether to include source citations
            retrieve_only: If True, only retrieve relevant chunks without LLM generation (faster for testing)
            expand_query: Whether to expand/rewrite the query using LLM (default: True)
            top_k: Number of chunks to retrieve (default: 10)
            
        Returns:
            Dictionary with 'answer' and optionally 'sources'
        """
        print(f"\nQuery: {question}")
        
        # Expand/rewrite query for better retrieval
        search_query = question
        if expand_query:
            search_query = self._expand_query(question)
        
        # Determine which queries to use for retrieval
        retrieval_queries = search_queries if search_queries else [search_query]

        if retrieve_only:
            # Just retrieve relevant chunks without LLM generation (fast - just embedding search)
            import time
            start = time.time()
            if len(retrieval_queries) > 1:
                nodes = self._retrieve_multi_query(retrieval_queries, top_k=top_k)
            else:
                nodes = self._retrieve_with_metadata_filtering(search_query, top_k=top_k)
            elapsed = time.time() - start
            print(f"✓ Embedding search completed in {elapsed:.3f} seconds (this is fast!)")
            return {
                'answer': f"Retrieved {len(nodes)} relevant chunks (retrieve_only mode)",
                'sources': [self._format_source(node) for node in nodes]
            }
        
        # Query the index (includes LLM generation - this is where timeouts happen)
        import time
        start = time.time()
        
        # First, do the fast embedding search with expanded query
        if len(retrieval_queries) > 1:
            retrieved_nodes = self._retrieve_multi_query(retrieval_queries, top_k=top_k)
        else:
            retrieved_nodes = self._retrieve_with_metadata_filtering(search_query, top_k=top_k)
        retrieval_time = time.time() - start
        print(f"\n✓ Retrieved {len(retrieved_nodes)} relevant chunks in {retrieval_time:.3f}s (embedding search)")
        print("\n" + "="*80)
        print("RETRIEVED CHUNKS (before LLM processing):")
        print("="*80)
        for i, node in enumerate(retrieved_nodes, 1):
            score = getattr(node, 'score', None)
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            identifier = metadata.get('identifier', 'N/A')
            heading = metadata.get('heading', 'N/A')
            tag = metadata.get('tag', 'N/A')
            
            # Format score properly - can't use conditional in format specifier
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"\n[{i}] Score: {score_str} | §{identifier} | {tag}")
            if heading and heading != 'N/A':
                print(f"    Heading: {heading}")
            print(f"    Text preview: {node.text[:200]}...")
            if len(node.text) > 200:
                print(f"    (Full text length: {len(node.text)} chars)")
        print("="*80)
        
        # Then generate answer with LLM using expanded query (this can be slow and cause timeouts)
        print("\nGenerating answer with LLM (this may take a while)...")
        try:
            # Use the expanded query for better context
            response = self.query_engine.query(search_query)
            
            total_time = time.time() - start
            llm_time = total_time - retrieval_time
            print(f"✓ Total time: {total_time:.2f}s (search: {retrieval_time:.3f}s, LLM: {llm_time:.2f}s)")
        except Exception as e:
            print(f"\n⚠️  LLM generation failed: {e}")
            print("But we successfully retrieved the chunks above - the embedding search is working!")
            # Return the retrieved chunks even if LLM fails
            return {
                'answer': f"LLM generation timed out, but retrieved {len(retrieved_nodes)} relevant chunks. See retrieved chunks above.",
                'sources': [self._format_source(node) for node in retrieved_nodes]
            }
        
        # Extract answer
        answer = str(response)
        
        result = {'answer': answer}
        
        # Add source citations if requested
        if include_sources and hasattr(response, 'source_nodes'):
            result['sources'] = [self._format_source(node, preview_length=200) for node in response.source_nodes[:5]]
        
        return result
    
    def rebuild_index(self, force: bool = False):
        """
        Rebuild the index from scratch.
        
        Args:
            force: If True, delete existing index without prompting
        """
        print("Rebuilding index...")
        # Remove existing index
        import shutil
        if self.index_dir.exists() and any(self.index_dir.iterdir()):
            if not force:
                print(f"⚠️  Existing index found at {self.index_dir}")
                response = input("Delete and rebuild? (y/N): ").strip().lower()
                if response != 'y':
                    print("Aborted.")
                    return
            shutil.rmtree(self.index_dir)
        
        # Build new index
        self.index = self._build_index()
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=20,
        )
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever,
        )
        print("Index rebuilt successfully")
    
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
        
        Args:
            ollama_base_url: Ollama server URL
            
        Returns:
            True if Ollama is accessible, False otherwise
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
    embedding_model: str = "nomic-embed-text",
    ollama_base_url: str = "http://localhost:11434",
    force_rebuild: bool = False
):
    """Build a vector index from a chunks file."""
    chunks_path = Path(chunks_file)
    if not chunks_path.exists():
        print(f"Error: Chunks file not found: {chunks_file}")
        return
    
    # Use TaxCodeRAG method to determine index directory
    index_dir = TaxCodeRAG.get_index_dir_from_chunks(chunks_file, index_name)
    
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
    
    # Initialize RAG system
    print("\nInitializing RAG system...")
    rag = TaxCodeRAG(
        chunks_path=str(chunks_path),
        index_dir=str(index_dir),
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url
    )
    
    # Check if we need to rebuild
    if force_rebuild or not index_dir.exists() or not any(index_dir.iterdir()):
        print("\nBuilding index (this may take a while)...")
        rag.rebuild_index(force=force_rebuild)
    else:
        print("\n✓ Index already exists and loaded successfully")
        print("Use --force to rebuild")
    
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
        python index.py build data/rag_chunks2.json
        python index.py build data/rag_chunks2.json --index-name my_index --force
        python index.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run', required=True)
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build a vector index from chunks')
    build_parser.add_argument('chunks_file', help='Path to chunks JSON file')
    build_parser.add_argument('--index-name', type=str, help='Name for the index directory')
    build_parser.add_argument('--embedding-model', type=str, default='nomic-embed-text', help='Ollama embedding model')
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

