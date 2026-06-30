from pathlib import Path
import shutil

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_PATH = Path("faiss_index")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

vector_store = None
retriever = None


def load_documents_from_source(file_path=None, raw_text=None):
    if raw_text and raw_text.strip():
        return [
            Document(
                page_content=raw_text.strip(),
                metadata={"source": "typed_input"},
            )
        ]

    if file_path:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            raise ValueError("Unsupported file type. Use .pdf, .txt, or .md")

        return loader.load()

    raise ValueError("Provide either raw_text or file_path.")


def ingest_data(file_path=None, raw_text=None):
    global vector_store, retriever

    docs = load_documents_from_source(file_path=file_path, raw_text=raw_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        return {
            "status": "error",
            "message": "No content could be extracted from the input."
        }

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_PATH))
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    return {
        "status": "success",
        "message": f"Indexed {len(chunks)} chunks successfully.",
        "chunks": len(chunks)
    }


def get_retriever():
    global retriever, vector_store

    if retriever is not None:
        return retriever

    if INDEX_PATH.exists():
        vector_store = FAISS.load_local(
            str(INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        return retriever

    return None


def reset_index():
    global vector_store, retriever

    vector_store = None
    retriever = None

    if INDEX_PATH.exists():
        shutil.rmtree(INDEX_PATH, ignore_errors=True)

    return {"status": "success", "message": "Index cleared."}