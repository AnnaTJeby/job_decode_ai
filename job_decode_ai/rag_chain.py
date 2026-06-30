import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

rag_chain = None
indexed_chunks = []


def _error_result(message: str):
    return {"status": "error", "message": message}


def _split_text(text: str, chunk_size: int = 700):
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []

    words = normalized.split()
    chunks = []
    current = []

    for word in words:
        if len(current) < chunk_size:
            current.append(word)
        else:
            chunks.append(" ".join(current))
            current = [word]

    if current:
        chunks.append(" ".join(current))

    return chunks


def _normalize(text: str):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _retrieve_context(question: str):
    if not indexed_chunks:
        return []

    query_tokens = set(_normalize(question))
    if not query_tokens:
        return indexed_chunks[:4]

    scored = []
    for index, chunk in enumerate(indexed_chunks):
        tokens = set(_normalize(chunk["text"]))
        score = len(query_tokens & tokens)
        if score > 0:
            scored.append((score, index, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return indexed_chunks[:4]

    return [chunk for _, _, chunk in scored[:4]]


def load_documents_from_source(file_path=None, raw_text=None):
    if raw_text and raw_text.strip():
        return [raw_text.strip()]

    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".txt":
            return [path.read_text(encoding="utf-8")]
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("PDF support requires pypdf. Install it with pip install pypdf") from exc

            reader = PdfReader(str(path))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            return [text]

        raise ValueError("Unsupported file type. Use .pdf or .txt")

    raise ValueError("Provide either raw_text or file_path.")


def build_chain(file_path=None, raw_text=None, temperature=0.0):
    global rag_chain, indexed_chunks

    try:
        documents = load_documents_from_source(file_path=file_path, raw_text=raw_text)
    except Exception as exc:
        return _error_result(f"Failed to load input: {exc}")

    chunks = []
    for doc in documents:
        chunks.extend(_split_text(doc))

    if not chunks:
        return _error_result("No content could be extracted from the input.")

    indexed_chunks = [{"text": chunk, "source": "typed_input" if raw_text and raw_text.strip() else "file"} for chunk in chunks]
    rag_chain = {"mode": "keyword"}

    return {"status": "success", "message": f"Indexed {len(chunks)} chunks successfully."}


def answer_question(question: str):
    if rag_chain is None:
        return {"answer": "Please load and index a job description first.", "sources": []}

    context_docs = _retrieve_context(question)
    if context_docs:
        best_chunk = context_docs[0]["text"]
        answer = best_chunk[:600]
    else:
        answer = "I can only answer questions about the provided job description. Please ask something related to it."

    sources = []
    for i, doc in enumerate(context_docs, 1):
        sources.append(
            {
                "chunk": i,
                "source": doc.get("source", "unknown"),
                "snippet": doc["text"][:300].replace("\n", " ").strip(),
            }
        )

    return {"answer": answer, "sources": sources}


def reset_index():
    global rag_chain, indexed_chunks
    rag_chain = None
    indexed_chunks = []

    index_dir = Path("faiss_index")
    if index_dir.exists():
        shutil.rmtree(index_dir)

    return {"status": "ok", "message": "Index cleared."}