"""
RAG Index Builder and Query Interface for Tax Code
Uses Ollama for embeddings and LLM, ChromaDB for vector storage.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import tiktoken
import chromadb
import ollama as _ollama
from rank_bm25 import BM25Okapi

COLLECTION_NAME = "tax_code_rag"

# nomic-embed-text task prefixes (asymmetric search); applied at embed time only,
# never stored in documents.
EMBED_DOC_PREFIX = "search_document: "
EMBED_QUERY_PREFIX = "search_query: "


def _tokenize(text: str) -> List[str]:
    """Tokenize for BM25: lowercase alphanumeric runs, so '§162(a)' matches '162 a'."""
    return re.findall(r'[a-z0-9]+', text.lower())


def _hierarchy_path(metadata: Dict) -> str:
    """Render a chunk's hierarchy chain as a readable path of headings,
    e.g. 'Income Taxes › Normal Taxes and Surtaxes › Computation of Taxable Income'."""
    hierarchy = metadata.get("hierarchy") or {}
    parts = []
    for level in ("subtitle", "chapter", "subchapter", "part", "subpart"):
        info = hierarchy.get(level) or {}
        heading = (info.get("heading") or "").strip()
        if heading:
            parts.append(heading)
    return " › ".join(parts)


def _log_llm_timing(response, label: str = "") -> None:
    """Print Ollama's prefill (prompt) vs decode (generation) breakdown for a call,
    so we can see whether time is spent reading context or writing the answer.
    Durations are nanoseconds; prompt fields are omitted when the prefill is cached."""
    ptok = getattr(response, "prompt_eval_count", None)
    pdur = getattr(response, "prompt_eval_duration", None) or 0
    otok = getattr(response, "eval_count", None)
    odur = getattr(response, "eval_duration", None) or 0
    prefill = f"{ptok} tok/{pdur / 1e9:.1f}s" if ptok else "cached"
    tps = (otok / (odur / 1e9)) if otok and odur else 0.0
    decode = f"{otok or 0} tok/{odur / 1e9:.1f}s ({tps:.1f} tok/s)"
    tag = f"{label} " if label else ""
    print(f"  llm {tag}: prefill {prefill} · decode {decode}")


class TaxCodeRAG:
    """RAG system for querying the US Tax Code."""

    MAX_EMBEDDING_TOKENS = 1800

    # Fixed context window applied to every generate() call. Ollama reloads the
    # model whenever num_ctx changes between calls, so a single shared value keeps
    # the model loaded once (with keep_alive) instead of reloading each step.
    LLM_NUM_CTX = 16384

    def __init__(
        self,
        chunks_path: Optional[str] = None,
        index_dir: Optional[str] = None,
        embedding_model: str = "nomic-custom",
        ollama_model: str = "gemma4:e2b",
        ollama_base_url: str = "http://localhost:11434",
        auto_build: bool = True,
    ):
        # set up the variables that store the path to the chunks and index
        data_dir = Path(__file__).parent / "data"
        self.chunks_path = Path(chunks_path) if chunks_path else data_dir / "rag_chunks2.json"
        self.index_dir = Path(index_dir) if index_dir else data_dir / "index_chroma"
        if not self.index_dir.is_absolute():
            self.index_dir = (Path(__file__).parent / self.index_dir).resolve()
        else:
            self.index_dir = self.index_dir.resolve()

        # initialize embedding model, LLM, encoding model, and ollama client
        self.embedding_model = embedding_model
        self.ollama_model = ollama_model
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        self._ollama = _ollama.Client(host=ollama_base_url)

        # create new directory for the chroma db storage, set up persistent client
        self.index_dir.mkdir(parents=True, exist_ok=True)
        print(f"Chroma DB path: {self.index_dir}")
        self.chroma_client = chromadb.PersistentClient(path=str(self.index_dir))
        self.collection = None
        self.bm25 = None
        self._bm25_chunks: List[Dict] = []
        self._section_map: Dict[str, List[Dict]] = {}

        if auto_build:
            self._init_index()

    def _init_index(self):
        """Load or build the collection and make it ready for queries."""
        self.collection = self._load_or_build_index()
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Build an in-memory BM25 index and a section-number → chunks map
        from the texts already stored in ChromaDB."""
        all_data = self.collection.get(include=["documents", "metadatas"])
        ids = all_data["ids"]
        docs = all_data["documents"]
        metadatas = [self._decode_meta(m) for m in all_data["metadatas"]]
        self._bm25_chunks = [
            {"id": ids[i], "text": docs[i], "metadata": metadatas[i]}
            for i in range(len(ids))
        ]
        self.bm25 = BM25Okapi([_tokenize(doc) for doc in docs])

        # exact-citation lookup: section number → list of chunks, in document order
        self._section_map: Dict[str, List[Dict]] = {}
        for chunk in self._bm25_chunks:
            # index by the bare section number ("162" for "162(a)"); dedupe since a
            # chunk's identifiers may list several subsections of the same section
            sections = set()
            for ident in chunk["metadata"].get("identifiers") or []:
                m = re.match(r'\d+[A-Z]{0,2}(?:-\d+)?', str(ident))
                if m:
                    sections.add(m.group(0))
            for sec in sections:
                self._section_map.setdefault(sec, []).append(chunk)
        print(f"BM25 index built ({len(docs)} documents, {len(self._section_map)} sections)")

    @staticmethod
    def _decode_meta(meta: Dict) -> Dict:
        """Decode JSON-encoded list fields from Chroma's flat metadata."""
        out = dict(meta or {})
        for key in ("identifiers", "ref_sections", "subsections"):
            val = out.get(key)
            if isinstance(val, str):
                try:
                    out[key] = json.loads(val)
                except json.JSONDecodeError:
                    out[key] = []
        return out

    def _load_chunks(self) -> List[Dict]:
        """Load chunks from the saved chunks_path"""
        print(f"Loading chunks from {self.chunks_path}")
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks")
        return chunks

    def _prepare_chunk_text(self, chunk: Dict) -> str:
        """Build prefixed, token-capped text for embedding."""
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        # get section/chapter numbers, create a prefix to the text including section # and heading
        ids = metadata.get("identifiers") or ( 
            [metadata["identifier"]] if metadata.get("identifier") is not None else []
        )
        prefix_parts = []
        if ids:
            prefix_parts.append("§" + ", §".join(str(i) for i in ids))
        if metadata.get("heading"):
            prefix_parts.append(metadata["heading"])
        # hierarchy context (chapter/subchapter/part headings) disambiguates
        # similarly-worded sections in different areas of the code
        hierarchy_path = metadata.get("hierarchy_path") or _hierarchy_path(metadata)
        if hierarchy_path:
            prefix_parts.append(f"[{hierarchy_path}]")
        prefix = f"{' '.join(prefix_parts)}: " if prefix_parts else ""

        # create the full text chunk and embed it with the token encoder
        full = prefix + text
        tokens = self.token_encoder.encode(full)
        if len(tokens) <= self.MAX_EMBEDDING_TOKENS:
            return full

        # truncate the full text if it's too long
        truncated = self.token_encoder.decode(tokens[: self.MAX_EMBEDDING_TOKENS])
        for delim in [".", "\n"]:
            pos = truncated.rfind(delim)
            if pos > len(truncated) * 0.9:
                return truncated[: pos + 1]
        return truncated

    def _build_index(self) -> chromadb.Collection:
        """Takes the chunks, adds metadata, embeds them, and stores in the chromadb collection"""
        print("Building index from chunks...")
        chunks = self._load_chunks()
        collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME)

        # take chunks in batches of 50, create the chunk w/ metadata
        BATCH = 50
        total = len(chunks)
        for start in range(0, total, BATCH):
            batch = chunks[start : start + BATCH]
            texts = [self._prepare_chunk_text(c) for c in batch]
            ids = [c["id"] for c in batch]
            metadatas = []
            for c in batch:
                m = c.get("metadata", {})
                identifiers = m.get("identifiers") or (
                    [m["identifier"]] if m.get("identifier") is not None else []
                )
                metadatas.append({
                    "identifier": str(m.get("identifier") or ""),
                    "identifiers": json.dumps([str(i) for i in identifiers if i is not None]),
                    "subsections": json.dumps(m.get("subsections") or []),
                    "heading": (m.get("heading") or "")[:100],
                    "tag": m.get("tag") or "",
                    "hierarchy_path": _hierarchy_path(m)[:300],
                    "parent_id": str(m.get("parent_id") or ""),
                    "ref_sections": json.dumps(m.get("ref_sections") or []),
                    "effective_date_note": (m.get("effective_date_note") or "")[:1200],
                    "token_count": int(m.get("token_count") or 0),
                })

            # embed the chunk (with the nomic document-task prefix) and add it to chroma db;
            # the stored document text stays unprefixed
            result = self._ollama.embed(
                model=self.embedding_model, input=[EMBED_DOC_PREFIX + t for t in texts]
            )
            collection.add(ids=ids, embeddings=result.embeddings, documents=texts, metadatas=metadatas)
            print(f"  Indexed {min(start + BATCH, total)}/{total} chunks", end="\r")

        print(f"\nIndex saved to Chroma ({self.index_dir})")
        return collection

    def _load_or_build_index(self) -> chromadb.Collection:
        collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME)
        if collection.count() == 0:
            print("No existing Chroma index. Building new index...")
            return self._build_index()
        print(f"Loaded existing Chroma index ({collection.count()} vectors)")
        return collection

    def retrieve(self, question: str, top_k: int = 10) -> List[Dict]:
        """Return top_k chunks using hybrid semantic + BM25 retrieval merged via RRF."""
        candidate_k = top_k * 3  # fetch more candidates from each source before merging

        # --- semantic search via ChromaDB ---
        embedded_query = self._ollama.embed(
            model=self.embedding_model, input=[EMBED_QUERY_PREFIX + question]
        )
        results = self.collection.query(
            query_embeddings=[embedded_query.embeddings[0]],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )
        semantic_hits: Dict[str, dict] = {}
        for i, doc in enumerate(results["documents"][0]):
            chunk_id = results["ids"][0][i]
            semantic_hits[chunk_id] = {
                "id": chunk_id,
                "text": doc,
                "score": 0.0,
                "metadata": self._decode_meta(results["metadatas"][0][i]),
            }

        # --- BM25 keyword search ---
        bm25_hits: Dict[str, dict] = {}
        if self.bm25 is not None:
            token_scores = self.bm25.get_scores(_tokenize(question))
            # get indices of top candidate_k scores
            top_indices = sorted(range(len(token_scores)), key=lambda i: token_scores[i], reverse=True)[:candidate_k]
            for idx in top_indices:
                chunk = self._bm25_chunks[idx]
                # chunk text came from Chroma documents, which are already prefixed
                bm25_hits[chunk["id"]] = {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "score": 0.0,
                    "metadata": chunk.get("metadata", {}),
                }

        # --- Reciprocal Rank Fusion ---
        # score = 1/(60 + rank) from each list; sum scores across lists
        RRF_K = 60
        rrf_scores: Dict[str, float] = {}

        for rank, chunk_id in enumerate(semantic_hits):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        for rank, chunk_id in enumerate(bm25_hits):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        # merge all candidates and sort by RRF score
        all_hits = {**bm25_hits, **semantic_hits}  # semantic overwrites text/meta if duplicate
        ranked = sorted(all_hits.keys(), key=lambda cid: rrf_scores.get(cid, 0.0), reverse=True)

        sources = []
        for chunk_id in ranked[:top_k]:
            hit = all_hits[chunk_id]
            hit["score"] = round(rrf_scores[chunk_id], 6)
            sources.append(hit)
        return sources

    def lookup_sections(self, section_numbers: List[str], max_chunks_per_section: int = 4) -> List[Dict]:
        """Exact-citation retrieval: return the chunks for the given IRC section
        numbers (bare numbers like '162' or '199A'), in document order."""
        sources = []
        for num in section_numbers:
            for chunk in self._section_map.get(str(num), [])[:max_chunks_per_section]:
                sources.append({
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "score": 0.0,
                    "metadata": chunk.get("metadata", {}),
                })
        return sources

    def expand_refs(self, sources: List[Dict], max_add: int = 0) -> List[Dict]:
        """One-hop cross-reference expansion: for the given sources, pull in the
        first chunk of each IRC section their text references (deduplicated
        against sources already present), up to max_add chunks."""
        have_ids = {s["id"] for s in sources}
        have_sections = set()
        for s in sources:
            for ident in s.get("metadata", {}).get("identifiers") or []:
                m = re.match(r'\d+[A-Z]{0,2}(?:-\d+)?', str(ident))
                if m:
                    have_sections.add(m.group(0))

        added = []
        for s in sources:
            for ref in s.get("metadata", {}).get("ref_sections") or []:
                if len(added) >= max_add:
                    return added
                if ref in have_sections:
                    continue
                chunks = self._section_map.get(str(ref), [])
                if chunks and chunks[0]["id"] not in have_ids:
                    chunk = chunks[0]
                    added.append({
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "score": 0.0,
                        "metadata": chunk.get("metadata", {}),
                    })
                    have_ids.add(chunk["id"])
                    have_sections.add(ref)
        return added

    def generate(self, prompt: str, options: Optional[dict] = None, label: str = "",
                 think: Optional[bool] = None) -> str:
        """Call the configured LLM with a prompt and return the response text.

        `think` controls the model's reasoning: gemma4:e4b is a reasoning model whose
        chain-of-thought is decoded before the answer (and billed as tokens). Pass
        think=False to skip it on steps where it adds no value (e.g. query expansion).
        """
        # Pin num_ctx across all calls so Ollama never reloads the model to resize
        # its context window (a reload costs ~20s and defeats keep_alive).
        merged_options = {"num_ctx": self.LLM_NUM_CTX, **(options or {})}
        response = self._ollama.chat(
            model=self.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options=merged_options,
            think=think,
            keep_alive=-1
        )
        _log_llm_timing(response, label)
        return response.message.content.strip()

    def format_source(self, source: Dict, preview_length: int = 200) -> Dict:
        text = source.get("text", "")
        return {
            "id": source["id"],
            "text": text,
            "text_preview": text[:preview_length] + "..." if len(text) > preview_length else text,
            "score": source.get("score"),
            "metadata": source.get("metadata", {}),
        }

    @staticmethod
    def get_index_dir_from_chunks(chunks_file: str, index_name: str = None) -> Path:
        data_dir = Path(__file__).parent / "data"
        if index_name:
            return data_dir / f"index_{index_name}"
        return data_dir / f"index_{Path(chunks_file).stem}"

    @staticmethod
    def check_ollama(ollama_base_url: str = "http://localhost:11434") -> bool:
        try:
            import requests
            return requests.get(f"{ollama_base_url}/api/tags", timeout=5).status_code == 200
        except Exception:
            return False


def list_indexes():
    """List all available indexes."""
    data_dir = Path(__file__).parent / "data"
    indexes = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("index_")]

    print("=" * 80)
    print("Available Indexes")
    print("=" * 80)
    if not indexes:
        print("No indexes found.")
        return
    for idx_dir in sorted(indexes):
        index_name = idx_dir.name.replace("index_", "")
        size_mb = sum(f.stat().st_size for f in idx_dir.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f"  {index_name:30s}  {size_mb:6.1f} MB  ({idx_dir})")
    print("=" * 80)


def build_index_cmd(
    chunks_file: str,
    index_name: str = None,
    embedding_model: str = "nomic-custom",
    ollama_base_url: str = "http://localhost:11434",
    force_rebuild: bool = False,
):
    """Build a vector index from a chunks file."""
    chunks_path = Path(chunks_file)
    if not chunks_path.exists():
        print(f"Error: Chunks file not found: {chunks_file}")
        return

    index_dir = TaxCodeRAG.get_index_dir_from_chunks(chunks_file, index_name).resolve()

    print("=" * 80)
    print(f"Chunks file:     {chunks_path}")
    print(f"Index directory: {index_dir}")
    print(f"Embedding model: {embedding_model}")
    print("=" * 80)

    if not TaxCodeRAG.check_ollama(ollama_base_url):
        print("Error: Could not connect to Ollama. Make sure it is running.")
        return

    rag = TaxCodeRAG(
        chunks_path=str(chunks_path),
        index_dir=str(index_dir),
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        auto_build=False,
    )

    existing_count = rag.chroma_client.get_or_create_collection(COLLECTION_NAME).count()
    if force_rebuild or existing_count == 0:
        if force_rebuild and existing_count > 0:
            print(f"\nForce-rebuilding (existing index has {existing_count} vectors)...")
        else:
            print("\nNo existing index found. Building...")
        try:
            rag.chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        rag._init_index()
    else:
        print(f"\n✓ Index already exists ({existing_count} vectors). Use --force to rebuild.")
        rag._init_index()

    print("\n" + "=" * 80)
    index_name_for_app = index_dir.name.replace("index_", "")
    print(f"Done. To use this index: --index-name {index_name_for_app}")
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tax Code RAG - Build and manage vector indexes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a vector index from chunks")
    build_parser.add_argument("chunks_file", nargs="?", default="data/rag_chunks2.json")
    build_parser.add_argument("--index-name", type=str, default="rag_chunks2")
    build_parser.add_argument("--embedding-model", type=str, default="nomic-custom")
    build_parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    build_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("list", help="List all available indexes")

    args = parser.parse_args()
    if args.command == "build":
        build_index_cmd(
            chunks_file=args.chunks_file,
            index_name=args.index_name,
            embedding_model=args.embedding_model,
            ollama_base_url=args.ollama_base_url,
            force_rebuild=args.force,
        )
    elif args.command == "list":
        list_indexes()


if __name__ == "__main__":
    main()
