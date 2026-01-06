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
        ollama_model: str = "llama3.1",
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the RAG system.
        
        Args:
            chunks_path: Path to rag_chunks.json file
            index_dir: Directory to save/load index (default: ./data/index)
            embedding_model: Ollama model name for embeddings (default: nomic-embed-text)
            ollama_model: Ollama model name for LLM (default: llama3.1)
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
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
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
        # Note: Embeddings are automatically persisted when index is saved
        print("Generating embeddings (this may take a while for large datasets)...")
        print("Embeddings will be saved and reused on subsequent runs.")
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        
        # Save index (this saves both the index structure AND all embeddings)
        print(f"\nSaving index and embeddings to {self.index_dir}")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(self.index_dir))
        print("Index and embeddings saved successfully!")
        print("Next time you run this, it will load the saved embeddings (much faster).")
        
        return index
    
    def _load_or_build_index(self) -> VectorStoreIndex:
        """Load existing index or build new one."""
        if self.index_dir.exists() and any(self.index_dir.iterdir()):
            try:
                print(f"Loading existing index and embeddings from {self.index_dir}")
                storage_context = StorageContext.from_defaults(persist_dir=str(self.index_dir))
                index = load_index_from_storage(storage_context)
                print("✓ Index and embeddings loaded successfully (no recalculation needed!)")
                return index
            except Exception as e:
                print(f"Failed to load index: {e}")
                print("Building new index...")
                return self._build_index()
        else:
            print("No existing index found. Building new index...")
            return self._build_index()
    
    def query(self, question: str, include_sources: bool = True) -> Dict:
        """
        Query the tax code.
        
        Args:
            question: User's question about the tax code
            include_sources: Whether to include source citations
            
        Returns:
            Dictionary with 'answer' and optionally 'sources'
        """
        print(f"\nQuery: {question}")
        
        # Query the index
        response = self.query_engine.query(question)
        
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
            response_mode="compact",
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

