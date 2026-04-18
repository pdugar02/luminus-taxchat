"""
RAG query logic for the Tax Code web app.
Provides get_rag and request handlers used by app.py.
"""

import time
from pathlib import Path

from rag import TaxCodeRAG

rag = None

EXPANSION_PROMPT = """
You are helping to improve a search query for finding information in the US Tax Code (Title 26).

Original question: {question}

Generate an improved search query that:
1. Includes key tax code terminology (e.g., "taxable income", "filing status", "tax brackets", "standard deduction")
2. Expands abbreviations (e.g., "MFS" -> "married filing separately")
3. Adds related terms (e.g., "tax rate" for "how much am I taxed")
4. Keeps the core intent of the question
5. If a specific section is asked for in the original query, make sure to include that section number
6. If there is something ambiguous, rephrase it with your best interpretation given the context.
7. Focus on terms that will actually appear in the relevant tax code sections.

Return ONLY the improved search query, nothing else:
"""

ANSWER_PROMPT = """You are an expert tax law assistant helping IRS employees find answers in the US Tax Code (Title 26).

Answer the following question using ONLY the tax code sections provided below. Quote section numbers when relevant. If the answer cannot be determined from the provided sections, say so clearly.

Tax Code Sections:
{context}

Question: {question}

Answer:"""


def get_rag(index_name: str = None, chunks_file: str = None) -> TaxCodeRAG:
    """Lazy initialization of the RAG system."""
    global rag
    if rag is None:
        data_dir = Path(__file__).parent / "data"

        if index_name:
            index_dir = data_dir / f"index_{index_name}"
        elif chunks_file:
            index_dir = data_dir / f"index_{Path(chunks_file).stem}"
        else:
            index_dir = data_dir / "index_rag_chunks2"
            chunks_file = str(data_dir / "rag_chunks2.json")

        rag = TaxCodeRAG(
            chunks_path=chunks_file,
            index_dir=str(index_dir),
        )
    return rag


def handle_query(data: dict) -> tuple[dict, int]:
    """Handle query requests and return (payload, status_code)."""
    question = data.get("question", "").strip()
    if not question:
        return {"error": "Question is required", "sources": []}, 400

    rag_system = get_rag()
    start = time.time()

    expanded = rag_system.generate(EXPANSION_PROMPT.format(question=question))
    sources = rag_system.retrieve(expanded, top_k=5)

    context = "\n\n".join(
        f"[§{s['metadata'].get('identifier', '?')}] {s['text']}"
        for s in sources
    )
    answer = rag_system.generate(ANSWER_PROMPT.format(context=context, question=question))

    print(f"Query took {time.time() - start:.1f}s")
    return {
        "answer": answer,
        "sources": [rag_system.format_source(s) for s in sources],
    }, 200
