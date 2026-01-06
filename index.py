"""
RAG Index Builder and Query Interface for Tax Code
Uses LlamaIndex with local Llama 3.1 via Ollama
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


class TaxCodeRAG:
    """RAG system for querying the US Tax Code."""
    
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
        self.chunks_path = chunks_path or Path(__file__).parent / "data" / "rag_chunks.json"
        self.index_dir = index_dir or Path(__file__).parent / "data" / "index"
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        
        # Initialize embedding model (using nomic-embed-text)
        print(f"Using Ollama for embeddings: {embedding_model}")
        Settings.embed_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url=ollama_base_url
        )
        
        # Initialize LLM (Ollama)
        print(f"Initializing Ollama LLM: {ollama_model}")
        Settings.llm = Ollama(model=ollama_model, base_url=ollama_base_url, request_timeout=120.0)
        
        # Load or build index
        self.index = self._load_or_build_index()
        
        # Use "default" response mode (faster, single LLM call) instead of "compact" (multiple calls)
        # The embedding search itself is fast - it's just cosine similarity on pre-computed vectors
        # Valid modes: "default", "compact", "tree_summarize", "refine", "simple_summarize"
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=10,
        )
    
    def _load_chunks(self) -> List[Dict]:
        """Load chunks from JSON file."""
        print(f"Loading chunks from {self.chunks_path}")
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks")
        return chunks
    
    def _chunk_to_node(self, chunk: Dict) -> TextNode:
        """Convert a chunk dictionary to a LlamaIndex TextNode."""
        # Build text with metadata context
        text = chunk.get('text', '')
        metadata = chunk.get('metadata', {})
        
        # Preserve all metadata from the chunk
        node_metadata = {
            'id': chunk.get('id'),
            'tag': metadata.get('tag'),
            'identifier': metadata.get('identifier'),
            'heading': metadata.get('heading'),
            'parent_id': metadata.get('parent_id'),
            'children_ids': metadata.get('children_ids', []),
            'element_id': metadata.get('element_id'),
            'chunk_index': metadata.get('chunk_index'),
        }
        
        # Include section/subsection identifiers in text for better context
        identifier_parts = []
        if metadata.get('identifier'):
            identifier_parts.append(f"§{metadata.get('identifier')}")
        if metadata.get('heading'):
            identifier_parts.append(metadata.get('heading'))
        
        if identifier_parts:
            text = f"{' '.join(identifier_parts)}: {text}"
        
        return TextNode(
            text=text,
            metadata=node_metadata,
            id_=chunk.get('id'),
        )
    
    def _build_index(self) -> VectorStoreIndex:
        """Build index from chunks."""
        print("Building index from chunks...")
        chunks = self._load_chunks()
        
        # Convert chunks to nodes
        nodes = [self._chunk_to_node(chunk) for chunk in chunks]
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
    
    def _load_or_build_index(self) -> VectorStoreIndex:
        """Load existing index or build new one."""
        if self.index_dir.exists() and any(self.index_dir.iterdir()):
            try:
                print(f"Loading existing index from {self.index_dir}")
                storage_context = StorageContext.from_defaults(persist_dir=str(self.index_dir))
                index = load_index_from_storage(storage_context)
                print("Index loaded successfully")
                return index
            except Exception as e:
                print(f"Failed to load index: {e}")
                print("Building new index...")
                return self._build_index()
        else:
            print("No existing index found. Building new index...")
            return self._build_index()
    
    def query(self, question: str, include_sources: bool = True, retrieve_only: bool = False) -> Dict:
        """
        Query the tax code.
        
        Args:
            question: User's question about the tax code
            include_sources: Whether to include source citations
            retrieve_only: If True, only retrieve relevant chunks without LLM generation (faster for testing)
            
        Returns:
            Dictionary with 'answer' and optionally 'sources'
        """
        print(f"\nQuery: {question}")
        
        if retrieve_only:
            # Just retrieve relevant chunks without LLM generation (fast - just embedding search)
            import time
            start = time.time()
            retriever = self.index.as_retriever(similarity_top_k=5)
            nodes = retriever.retrieve(question)
            elapsed = time.time() - start
            print(f"✓ Embedding search completed in {elapsed:.3f} seconds (this is fast!)")
            return {
                'answer': f"Retrieved {len(nodes)} relevant chunks (retrieve_only mode)",
                'sources': [
                    {
                        'id': node.node_id,
                        'text_preview': node.text[:300] + '...' if len(node.text) > 300 else node.text,
                        'metadata': node.metadata if hasattr(node, 'metadata') else {},
                        'score': getattr(node, 'score', None)
                    }
                    for node in nodes
                ]
            }
        
        # Query the index (includes LLM generation - this is where timeouts happen)
        import time
        start = time.time()
        
        # First, do the fast embedding search
        retriever = self.index.as_retriever(similarity_top_k=5)
        retrieved_nodes = retriever.retrieve(question)
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
        
        # Then generate answer with LLM (this can be slow and cause timeouts)
        print("\nGenerating answer with LLM (this may take a while)...")
        try:
            response = self.query_engine.query(question)
            
            total_time = time.time() - start
            llm_time = total_time - retrieval_time
            print(f"✓ Total time: {total_time:.2f}s (search: {retrieval_time:.3f}s, LLM: {llm_time:.2f}s)")
        except Exception as e:
            print(f"\n⚠️  LLM generation failed: {e}")
            print("But we successfully retrieved the chunks above - the embedding search is working!")
            # Return the retrieved chunks even if LLM fails
            return {
                'answer': f"LLM generation timed out, but retrieved {len(retrieved_nodes)} relevant chunks. See retrieved chunks above.",
                'sources': [
                    {
                        'id': node.node_id,
                        'text_preview': node.text[:300] + '...' if len(node.text) > 300 else node.text,
                        'score': getattr(node, 'score', None),
                        'metadata': node.metadata if hasattr(node, 'metadata') else {}
                    }
                    for node in retrieved_nodes
                ]
            }
        
        # Extract answer
        answer = str(response)
        
        result = {'answer': answer}
        
        # Add source citations if requested
        if include_sources and hasattr(response, 'source_nodes'):
            sources = []
            for node in response.source_nodes[:5]:  # Top 5 sources
                source_info = {
                    'id': node.node_id,
                    'text_preview': node.text[:200] + '...' if len(node.text) > 200 else node.text,
                }
                # Add metadata if available
                if hasattr(node, 'metadata') and node.metadata:
                    source_info['metadata'] = {
                        'identifier': node.metadata.get('identifier'),
                        'heading': node.metadata.get('heading'),
                        'tag': node.metadata.get('tag'),
                    }
                sources.append(source_info)
            result['sources'] = sources
        
        return result
    
    def rebuild_index(self):
        """Rebuild the index from scratch."""
        print("Rebuilding index...")
        # Remove existing index
        if self.index_dir.exists():
            import shutil
            shutil.rmtree(self.index_dir)
        # Build new index
        self.index = self._build_index()
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
        )
        print("Index rebuilt successfully")


def main():
    """Main function for interactive querying."""
    import sys
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("Warning: Ollama may not be running. Make sure Ollama is installed and running.")
            print("Install: https://ollama.ai")
            print("Then run: ollama pull llama3.1")
    except Exception as e:
        print(f"Warning: Could not connect to Ollama: {e}")
        print("Make sure Ollama is installed and running.")
        print("Install: https://ollama.ai")
        print("Then run: ollama pull llama3.1")
    
    # Initialize RAG system
    # Uses nomic-embed-text for embeddings (install with: ollama pull nomic-embed-text)
    rag = TaxCodeRAG()
    
    # Interactive mode
    if len(sys.argv) > 1:
        # Single query from command line
        question = ' '.join(sys.argv[1:])
        result = rag.query(question)
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result['answer'])
        if 'sources' in result:
            print("\n" + "="*80)
            print("SOURCES:")
            print("="*80)
            for i, source in enumerate(result['sources'], 1):
                print(f"\n{i}. {source.get('metadata', {}).get('identifier', 'N/A')} - {source.get('metadata', {}).get('heading', 'N/A')}")
                print(f"   Preview: {source['text_preview']}")
    else:
        # Interactive mode
        print("\n" + "="*80)
        print("Tax Code RAG System")
        print("="*80)
        print("Type your questions about the tax code. Type 'quit' or 'exit' to exit.\n")
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                result = rag.query(question)
                print("\n" + "="*80)
                print("ANSWER:")
                print("="*80)
                print(result['answer'])
                
                if 'sources' in result:
                    print("\n" + "="*80)
                    print("SOURCES:")
                    print("="*80)
                    for i, source in enumerate(result['sources'], 1):
                        meta = source.get('metadata', {})
                        identifier = meta.get('identifier', 'N/A')
                        heading = meta.get('heading', 'N/A')
                        print(f"\n{i}. §{identifier} - {heading}")
                        print(f"   Preview: {source['text_preview']}")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()

